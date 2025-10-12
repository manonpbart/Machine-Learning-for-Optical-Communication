import numpy as np

def gauss(cfg):
    """
    Generate a Gaussian beam using parameters from a Config object.
    """
    X, Y = cfg.grid()
    R = np.sqrt(X**2 + Y**2)
    return np.exp(-R**2 / cfg.w0**2)