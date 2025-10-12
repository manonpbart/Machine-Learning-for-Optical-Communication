from .config import Config as _BaseConfig
from .beams.gaussian import gauss as _gauss_base
from .beams.lg import lg as _lg_base
from .beams.hg import hg as _hg_base
from .beams.ig import ig as _ig_base
from .propagation.propagation import propagation as _prop_base
from .propagation.turbulence import turbulence as _turb_base

# --- Global configuration---
_default_config = None

def get_default_config():
    """Return the current global configuration."""
    global _default_config
    if _default_config is None:
        _default_config = Config()  #uses the class defined below
    return _default_config


def set_default_config(cfg):
    """Manually set a new global configuration."""
    global _default_config
    _default_config = cfg


class Config(_BaseConfig):
    """
    Global beam configuration class.
    Creating a new Config automatically sets it as the package default.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        set_default_config(self)  #update global default when created

    def __repr__(self):
        return super().__repr__()


# --- Ensure beam and propagation wrappers use current config automatically ---
def gauss(cfg=None, **kwargs):
    cfg = cfg or get_default_config()
    return _gauss_base(cfg=cfg, **kwargs)


def lg(p, l, cfg=None, **kwargs):
    cfg = cfg or get_default_config()
    return _lg_base(p=p, l=l, cfg=cfg, **kwargs)


def hg(n, m, cfg=None, **kwargs):
    cfg = cfg or get_default_config()
    return _hg_base(n=n, m=m, cfg=cfg, **kwargs)


def ig(p, m, beam="e", cfg=None, **kwargs):
    cfg = cfg or get_default_config()
    return _ig_base(p=p, m=m, beam=beam, cfg=cfg, **kwargs)


def propagation(E, z=None, cfg=None, **kwargs):
    cfg = cfg or get_default_config()
    z = z or cfg.z_default
    return _prop_base(E, z=z, cfg=cfg, **kwargs)


def turbulence(cfg=None, **kwargs):
    cfg = cfg or get_default_config()
    return _turb_base(cfg=cfg, **kwargs)
