import numpy as np
from numpy.fft import fftshift, ifftshift, ifft2

def iFT(X):
    return ifftshift(ifft2(fftshift(X))) 

def turbulence(cfg, Cn2=None, l_max=25, l_min=1e-3):
    #Default parameters
    Cn2 = Cn2 or cfg.Cn2
    l_max = l_max or cfg.l_max
    l_min = l_min or cfg.l_min
    Z = 100.0  # m

    k = 2 * np.pi / cfg.wavelength
    r0 = (0.423 * (k**2) * Cn2 * Z) ** (-3/5) #Freid Parameter

    #Frequency Grid
    fx = np.fft.fftshift(np.fft.fftfreq(cfg.size_x, cfg.pixel_size))
    fy = np.fft.fftshift(np.fft.fftfreq(cfg.size_y, cfg.pixel_size))
    FX, FY = np.meshgrid(fx, fy)
    f = np.sqrt(FX**2 + FY**2)

    fm = 5.92 / (2 * np.pi * l_min)
    f0 = 1 / l_max
    PSD_phi = 0.023 * r0**(-5/3) * np.exp(-(f / fm)**2) / ((f**2 + f0**2)**(11/6))
    PSD_phi[cfg.size_y//2, cfg.size_x//2] = 0.0

    delta_fx = 1.0 / (cfg.size_x * cfg.pixel_size)
    delta_fy = 1.0 / (cfg.size_y * cfg.pixel_size)

    rand_spec = (np.random.normal(size=PSD_phi.shape) +
                 1j * np.random.normal(size=PSD_phi.shape))

    cn = rand_spec * np.sqrt(PSD_phi) * np.sqrt(delta_fx * delta_fy)
    phi_xy = np.real(np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(cn)))) * (cfg.size_x * cfg.size_y)

    return phi_xy

