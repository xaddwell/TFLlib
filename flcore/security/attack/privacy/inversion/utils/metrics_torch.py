import torch
import lpips

def cal_lpips(rec_data, GT_data, net='vgg'):
    if net not in ['vgg', 'alexnet']:
        print(f"lpips for {net} is not supported")
        return None
    lpips_loss = lpips.LPIPS(net=net, spatial=False)
    lpips_score = lpips_loss(rec_data, GT_data).squeeze().mean()
    return lpips_score

def psnr_compute(img_batch, ref_batch, batched=True, factor=1.0, clip=False):
    """Standard PSNR."""
    if clip:
        img_batch = torch.clamp(img_batch, 0, 1)

    if batched:
        mse = ((img_batch.detach() - ref_batch) ** 2).mean()
        if mse > 0 and torch.isfinite(mse):
            return 10 * torch.log10(factor ** 2 / mse)
        elif not torch.isfinite(mse):
            return [torch.tensor(float("nan"), device=img_batch.device)] * 2
        else:
            return [torch.tensor(float("inf"), device=img_batch.device)] * 2
    else:
        B = img_batch.shape[0]
        mse_per_example = ((img_batch.detach() - ref_batch) ** 2).view(B, -1).mean(dim=1)
        if any(mse_per_example == 0):
            return [torch.tensor(float("inf"), device=img_batch.device)] * 2
        elif not all(torch.isfinite(mse_per_example)):
            return [torch.tensor(float("nan"), device=img_batch.device)] * 2
        else:
            psnr_per_example = 10 * torch.log10(factor ** 2 / mse_per_example)
            return psnr_per_example.mean().item(), psnr_per_example.max().item()

def cw_ssim(img_batch, ref_batch, scales=5, skip_scales=None, K=1e-6, reduction="mean"):
    """Batched complex wavelet structural similarity.

    As in Zhou Wang and Eero P. Simoncelli, "TRANSLATION INSENSITIVE IMAGE SIMILARITY IN COMPLEX WAVELET DOMAIN"
    Ok, not quite, this implementation computes no local SSIM and neither averaging over local patches and uses only
    the existing wavelet structure to provide a similar scale-invariant decomposition.

    skip_scales can be a list like [True, False, False, False] marking levels to be skipped.
    K is a small fudge factor.
    """
    try:
        from pytorch_wavelets import DTCWTForward
    except ModuleNotFoundError:
        print(
            "To utilize wavelet SSIM, install pytorch wavelets from https://github.com/fbcotter/pytorch_wavelets."
        )
        return torch.as_tensor(float("NaN")), torch.as_tensor(float("NaN"))

    # 1) Compute wavelets:
    setup = dict(device=img_batch.device, dtype=img_batch.dtype)
    if skip_scales is not None:
        include_scale = [~s for s in skip_scales]
        total_scales = scales - sum(skip_scales)
    else:
        include_scale = True
        total_scales = scales
    xfm = DTCWTForward(J=scales, biort="near_sym_b", qshift="qshift_b", include_scale=include_scale).to(**setup)
    img_coefficients = xfm(img_batch)
    ref_coefficients = xfm(ref_batch)

    # 2) Multiscale complex SSIM:
    ssim = 0
    for xs, ys in zip(img_coefficients[1], ref_coefficients[1]):
        if len(xs) > 0:
            xc = torch.view_as_complex(xs)
            yc = torch.view_as_complex(ys)

            conj_product = (xc * yc.conj()).sum(dim=2).abs()
            square_img = (xc * xc.conj()).abs().sum(dim=2)
            square_ref = (yc * yc.conj()).abs().sum(dim=2)

            ssim_val = (2 * conj_product + K) / (square_img + square_ref + K)
            ssim += ssim_val.mean(dim=[1, 2, 3])
    ssim = ssim / total_scales
    return ssim.mean().item(), ssim.max().item()