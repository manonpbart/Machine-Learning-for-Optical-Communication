#beam_simulation/config.py
import numpy as np

class Config:
    """
    Configuration object for beam parameters.
    This acts as the global setup for the entire simulation suite.
    """

    def __init__(
        self,
        wavelength=810e-9,
        size_x=300,
        size_y=300,
        pixel_size=8e-6,
        w0=0.45e-3,
        z_default=1e-2,
        Cn2=3e-13,
        l_max=1e-1,
        l_min=1e-3,
    ):
        self.wavelength = wavelength
        self.size_x = size_x
        self.size_y = size_y
        self.pixel_size = pixel_size
        self.w0 = w0
        self.z_default = z_default
        self.Cn2 = Cn2
        self.l_max = l_max
        self.l_min = l_min

    def grid(self):
        x = (np.arange(-self.size_x/2, self.size_x/2) * self.pixel_size)
        y = (np.arange(-self.size_y/2, self.size_y/2) * self.pixel_size)
        X, Y = np.meshgrid(x, y)
        return X, Y

    def __repr__(self):
        """Readable string for printing the configuration."""
        return (
            f"Beam Simulation Configuration:\n"
            f"  Wavelength: {self.wavelength*1e9:.1f} nm\n"
            f"  Beam waist (w0): {self.w0*1e3:.3f} mm\n"
            f"  Size: {self.size_x} × {self.size_y} px\n"
            f"  Pixel size: {self.pixel_size*1e6:.2f} µm\n"
        )

# ------------------------------
# Global Config Handling
# ------------------------------

_default_config = Config()


def get_default_config():
    """Returns current default configuration."""
    global _default_config
    return _default_config


def set_default_config(cfg):
    """Sets new default configuration for all beams."""
    global _default_config
    _default_config = cfg


