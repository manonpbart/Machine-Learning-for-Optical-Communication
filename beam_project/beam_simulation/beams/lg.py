import numpy as np, math
from scipy.special import eval_genlaguerre

def lg(p, l, cfg, z=None):
    """
    Generate complex Laguerre-Gaussian beam LG_p^l(x,y)
    """
    if z is None:
        z = cfg.z_default

    X, Y = cfg.grid()
    r, theta = np.sqrt(X**2 + Y**2), np.arctan2(Y, X)

    k = 2 * np.pi / cfg.wavelength
    zR = k * cfg.w0**2 / 2
    phiZ = np.arctan(z / zR)
    Rz = z * (1 + (zR / z)**2)
    w = cfg.w0 * np.sqrt(1 + (z / zR)**2)
    normC = np.sqrt((2 * math.factorial(p)) / (np.pi * math.factorial(p + abs(l))))
    laguerre = eval_genlaguerre(p, abs(l), 2 * r**2 / cfg.w0**2)

    amp = normC * (1/cfg.w0) * (np.sqrt(2) * r / cfg.w0)**abs(l) * np.exp(-r**2 / cfg.w0**2) * laguerre
    phase = (-l * theta) - (k * r**2 / (2 * Rz)) + phiZ
    return amp * np.exp(1j * phase)