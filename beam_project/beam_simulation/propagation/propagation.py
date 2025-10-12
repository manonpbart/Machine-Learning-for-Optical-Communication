import numpy as np

def FT(x):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))

def iFT(x):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x)))

def propagation(input_beam, z, cfg):
    """
    Propagate an optical field using angular spectrum method.
    """
    size_y, size_x = input_beam.shape
    fx = np.fft.fftshift(np.fft.fftfreq(size_x, cfg.pixel_size))
    fy = np.fft.fftshift(np.fft.fftfreq(size_y, cfg.pixel_size))
    FX, FY = np.meshgrid(fx, fy)
    H = np.exp(-1j * np.pi * cfg.wavelength * z * (FX**2 + FY**2))
    arg = (2 * np.pi)**2 * ((1. / cfg.wavelength) ** 2 - FX**2 - FY**2)
    tmp = np.sqrt(np.abs(arg))
    kz = tmp #np.where(arg >= 0, tmp, 1j*tmp) #this gets rid of evanescent waves

    H = np.exp(1j * z * kz) 
    return iFT(FT(input_beam) * H)