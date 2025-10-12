"""
Hermite-Gaussian beam generator.
"""

import numpy as np
import math
from scipy.special import hermite

def hg(n, m, cfg, z=None):
    """
    Generate Hermite-Gaussian (HG_nm) beam.

    Parameters
    ----------
    n, m : int
        Mode indices (order of Hermite polynomial).
    cfg : Config
        Configuration object containing beam parameters.
    z : float, optional
        Propagation distance [m]. Defaults to cfg.z_default.
        
    Returns
    -------
    np.ndarray
        Complex field of HG_nm beam.
    """
    if z is None:
        z = cfg.z_default

    X, Y = cfg.grid()
    k = 2 * np.pi / cfg.wavelength
    zR = k * cfg.w0**2 / 2
    phiZ = np.arctan(z / zR)
    w = cfg.w0 * np.sqrt(1 + (z / zR)**2)
    Rz = z * (1 + (zR / z)**2)

    Hn = hermite(n)(np.sqrt(2) * X / w)
    Hm = hermite(m)(np.sqrt(2) * Y / w)

    amp = Hn * Hm * np.exp(-(X**2 + Y**2) / w**2)
    phase = np.exp(1j * ((k * (X**2 + Y**2) / (2 * Rz)) - (n + m + 1) * phiZ))
    norm = np.sqrt(2 / (np.pi * w**2 * (2**(n + m)) * math.factorial(n) * math.factorial(m)))
    return norm * amp * phase