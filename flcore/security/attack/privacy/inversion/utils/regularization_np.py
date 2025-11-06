import numpy as np

def TV(data, scale):
    dx = np.mean(np.abs(data[:, :, :, :-1] - np[:, :, :, 1:]))
    dy = np.mean(np.abs(data[:, :, :-1, :] - np[:, :, 1:, :]))
    return (dx + dy) * scale

def Lp(data, pnorm, scale):
    return 1 / pnorm * np.mean(np.pow(data, pnorm)) * scale

def Deep_inversion():
    pass

RegFn_list = ['TV', 'Lp', 'Deep_inversion']