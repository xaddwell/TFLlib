import numpy as np
import lpips

def mse(rec_data, GT_data):
    mse_score = np.mean(np.power(rec_data-GT_data, 2), axis=[1, 2, 3])
    avg_mse = np.mean(mse_score).item()
    return avg_mse, mse_score

def psnr(rec_data, GT_data, factor=255):
    # zm: unsure
    mse_score, avg_mse =  mse(rec_data, GT_data)
    psnr_score = 20 * np.log10(factor*1.0 / mse_score)
    avg_psnr = 20 * np.log10(factor*1.0 / avg_mse)
    if avg_mse > 0 and np.isfinite(avg_mse):
        return psnr_score, avg_psnr
    elif not np.isfinite(avg_mse):
        return np.array([np.nan, np.nan])
    else:
        return np.array([np.inf, np.inf])
    
def lpips(rec_data, GT_data, net='vgg'):
    lpips_loss = lpips.LPIPS(net=net, spatial=False)
    lpips_score = lpips_loss(rec_data, GT_data).squeeze().mean()
    return lpips_score
