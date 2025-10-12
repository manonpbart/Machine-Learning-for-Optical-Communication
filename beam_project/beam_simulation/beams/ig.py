import numpy as np
from numpy import linalg as la
from scipy.special import gamma
from beam_simulation.config import get_default_config

def ig(p, m, beam="e", elliptic_param=None, cfg=None):
    """
    Calculate even or odd Ince–Gaussian beams

    Parameters
    ----------
    p : int
        Order of the Ince polynomial.
    m : int
        Mode index.
    beam : str
        "e" for even or "o" for odd mode.
    elliptic_param : float, optional
        Optional elliptic parameter q = (2*f0^2) / w0^2.
        If not provided, defaults to the original a/b pair.
    cfg : Config, optional
        Beam configuration; if None, uses global config.

    Returns
    -------
    np.ndarray
        Complex Ince–Gaussian field.
    """
    cfg = cfg or get_default_config()
    size = cfg.size_x
    pixel_size = cfg.pixel_size
    w0 = cfg.w0
    wavelength = cfg.wavelength
    z = cfg.z_default

    #Cartesian grid
    X = np.arange(-size / 2, size / 2) * pixel_size
    Y = np.arange(-size / 2, size / 2) * pixel_size
    x, y = np.meshgrid(X, Y)
    rr = np.sqrt(x**2 + y**2)
    k = 2 * np.pi / wavelength
    z0 = k * w0**2 / 2
    w = w0 * np.sqrt(1 + (z / z0) ** 2)

    #Elliptic parameter setup
    if elliptic_param is None:
        a = 0.3100000175
        b = 0.3
        f0 = np.sqrt(a**2 - b**2)
        elliptic_param = (2 * f0**2) / w0**2
    else:
        f0 = np.sqrt((elliptic_param * w0**2) / 2)
        a = np.sqrt(f0**2 + (0.3**2))  # maintain aspect ratio roughly
        b = 0.3

    #Elliptical coordinate transform
    def elliptical(b):
        N = size
        c2 = (a**2 - b**2) * (w0 / w) ** 2
        c = 2 * c2
        x2, y2 = x**2, y**2
        B = x2 + y2 - c2
        del2 = B**2 + 2 * c * y2
        del1 = np.sqrt(del2)
        p = (-B + del1) / c
        p = np.where(p > 1, 1, p)
        p = np.sqrt(p)
        et0 = np.arcsin(p)
        eta = np.zeros_like(x)
        qv = (-B - del1) / c
        del2q = qv**2 - qv
        del1q = np.sqrt(del2q)
        zeta = np.log(1 - 2 * qv + 2 * del1q) / 2

        for i in range(N):
            for j in range(N):
                if x[i, j] >= 0 and y[i, j] >= 0:
                    eta[i, j] = et0[i, j]
                elif x[i, j] < 0 and y[i, j] >= 0:
                    eta[i, j] = np.pi - et0[i, j]
                elif x[i, j] <= 0 and y[i, j] < 0:
                    eta[i, j] = np.pi + et0[i, j]
                elif x[i, j] > 0 and y[i, j] < 0:
                    eta[i, j] = 2 * np.pi - et0[i, j]
        return f0, eta, zeta

    f0, eta, zeta = elliptical(b)

    #IG definition
    def ig_internal(p, m, beam):
        q = (2 * f0**2) / (w0**2)
        z1 = np.transpose(eta)
        z2 = np.transpose(1j * zeta)
        c1, c2 = z1.shape

        if ((-1) ** (m - p)) != 1:
            print("ERROR: p and m do not have same parity")
        if m < 1 or m > p:
            print("ERROR: Invalid m range")

        if beam == "e":
            if p % 2 == 0:
                j = p // 2
                N = int(j + 1)
                n = int(m / 2 + 1)
                m1, m2, m3 = [], [], []
                m2.insert(0, 2 * q * j)
                m3.insert(0, 0)
                for i in range(1, N):
                    m1.append(q * (j + i))
                m1 = np.diag(m1, 1)
                for i in range(1, N - 1):
                    m2.append(q * (j - i))
                m2 = np.diag(m2, -1)
                for i in range(0, N - 1):
                    m3.append(4 * (i + 1) ** 2)
                m3 = np.diag(m3)
                M = m1 + m2 + m3
                ets, A = la.eig(M)
                idx = np.argsort(ets)
                ets = np.sort(ets)
                A = A[:, idx]
                mv = np.arange(2, p + 1, 2)
                N22 = [A[i, n - 1] for i in range(1, int(p / 2 + 1))]
                N2 = np.sqrt(
                    A[0, n - 1] ** 2 * 2 * gamma(p / 2 + 1) ** 2
                    + np.sum(
                        (
                            np.sqrt(
                                gamma((p + mv) / 2 + 1)
                                * gamma((p - mv) / 2 + 1)
                            )
                            * N22
                        )
                        ** 2
                    )
                )
                NS = np.sign(np.sum(A[:, n - 1]))
                A = A / N2 * NS
                r = np.arange(0, N, 1)
                R, X = np.meshgrid(r, z1)
                IP1 = np.dot(np.cos(2 * X * R), (A[:, n - 1].reshape(N, 1)))
                IP1 = np.transpose(IP1.reshape(c1, c2))
                R1, X1 = np.meshgrid(r, z2)
                IP2 = np.dot(np.cos(2 * X1 * R1), (A[:, n - 1].reshape(N, 1)))
                IP2 = np.transpose(IP2.reshape(c1, c2))
                R2, X2 = np.meshgrid(r, 0)
                IP3 = np.dot(np.cos(2 * X2 * R2), (A[:, n - 1].reshape(N, 1)))
                R4, X4 = np.meshgrid(r, np.pi / 2)
                IP4 = np.dot(np.cos(2 * X4 * R4), (A[:, n - 1].reshape(N, 1)))
                Norm = (
                    (-1) ** (m / 2)
                    * np.sqrt(2)
                    * gamma(p / 2 + 1)
                    * A[0, n - 1]
                    * np.sqrt(2 / np.pi)
                    / w0
                    / IP3
                    / IP4
                )
            else:
                print("Odd p not supported")
                IP1 = IP2 = Norm = np.zeros_like(z1)
        else:
            print("TO DO: Odd Ince–Gaussian not yet implemented here.")
            IP1 = IP2 = Norm = np.zeros_like(z1)

        return IP1, IP2, Norm

    IP1, IP2, Norm = ig_internal(p, m, beam)
    add = (
        Norm
        * (IP1 * IP2 * np.exp(-(rr / w0) ** 2))
        * (
            w0
            * np.exp(1j * k * z)
            * np.exp((1j * k * z * rr**2) / (2 * (z**2 + z0**2)))
            * np.exp(-1j * (p + 1) * np.arctan(z / z0))
        )
        / w
    )

    return add
