"""
********************************************************************************
operators
********************************************************************************
"""

import numpy as np

def advection(u, v, f, dx, dy, dt, scheme="KK"):
    """
    advect f with advection rate u, v
    """
    if scheme == "1st":
        # 1st order derivatives
        f_x = (f[3:-1, 2:-2] - f[1:-3, 2:-2]) / (2. * dx)
        f_y = (f[2:-2, 3:-1] - f[2:-2, 1:-3]) / (2. * dy)

        # 2nd order derivatives
        f_xx = (f[3:-1, 2:-2] - 2. * f[2:-2, 2:-2] + f[1:-3, 2:-2]) / dx**2
        f_yy = (f[2:-2, 3:-1] - 2. * f[2:-2, 2:-2] + f[2:-2, 1:-3]) / dy**2

        # advection
        advc = (u[2:-2, 2:-2] * f_x - np.abs(u[2:-2, 2:-2]) * dx / 2. * f_xx) \
                + (v[2:-2, 2:-2] * f_y - np.abs(v[2:-2, 2:-2]) * dy / 2. * f_yy)

    elif scheme == "LW":
        # 1st order derivatives
        f_x = (f[3:-1, 2:-2] - f[1:-3, 2:-2]) / (2. * dx)
        f_y = (f[2:-2, 3:-1] - f[2:-2, 1:-3]) / (2. * dy)

        # 2nd order derivatives
        f_xx = (f[3:-1, 2:-2] - 2. * f[2:-2, 2:-2] + f[1:-3, 2:-2]) / dx**2
        f_yy = (f[2:-2, 3:-1] - 2. * f[2:-2, 2:-2] + f[2:-2, 1:-3]) / dy**2

        # advection
        advc = (u[2:-2, 2:-2] * f_x - np.abs(u[2:-2, 2:-2])**2 * dt / 2. * f_xx) \
                + (v[2:-2, 2:-2] * f_y - np.abs(v[2:-2, 2:-2])**2 * dt / 2. * f_yy)

    elif scheme == "QUICK":
        # 1st order derivatives
        f_x = (- f[4:, 2:-2] + 10. * f[3:-1, 2:-2] - 10. * f[1:-3, 2:-2] + f[:-4, 2:-2]) / (16. * dx)
        f_y = (- f[2:-2, 4:] + 10. * f[2:-2, 3:-1] - 10. * f[2:-2, 1:-3] + f[2:-2, :-4]) / (16. * dy)

        # 4th order derivatives
        f_xxxx = (- f[4:, 2:-2] + 4. * f[3:-1, 2:-2] - 6. * f[2:-2, 2:-2] + 4. * f[1:-3, 2:-2] - f[:-4, 2:-2]) / dx**4
        f_yyyy = (- f[2:-2, 4:] + 4. * f[2:-2, 3:-1] - 6. * f[2:-2, 2:-2] + 4. * f[2:-2, 1:-3] - f[2:-2, :-4]) / dy**4

        # advection
        advc = (u[2:-2, 2:-2] * f_x - np.abs(u[2:-2, 2:-2]) * dx**3 / 16. * f_xxxx) \
                + (v[2:-2, 2:-2] * f_y - np.abs(v[2:-2, 2:-2]) * dy**3 / 16. * f_yyyy)

    elif scheme == "QUICKEST":
        raise NotImplementedError

    elif scheme == "KK":
        # 1st order derivatives
        f_x = (- f[4:, 2:-2] + 8. * f[3:-1, 2:-2] - 8. * f[1:-3, 2:-2] + f[:-4, 2:-2]) / (12. * dx)
        f_y = (- f[2:-2, 4:] + 8. * f[2:-2, 3:-1] - 8. * f[2:-2, 1:-3] + f[2:-2, :-4]) / (12. * dy)

        # 4th order derivatives
        f_xxxx = (f[4:, 2:-2] - 4. * f[3:-1, 2:-2] + 6. * f[2:-2, 2:-2] - 4. * f[1:-3, 2:-2] + f[:-4, 2:-2]) / dx**4
        f_yyyy = (f[2:-2, 4:] - 4. * f[2:-2, 3:-1] + 6. * f[2:-2, 2:-2] - 4. * f[2:-2, 1:-3] + f[2:-2, :-4]) / dy**4

        # advection
        advc = (u[2:-2, 2:-2] * f_x + np.abs(u[2:-2, 2:-2]) * dx**3 / 4. * f_xxxx) \
                + (v[2:-2, 2:-2] * f_y + np.abs(v[2:-2, 2:-2]) * dy**3 / 4. * f_yyyy)

    return advc


def diffusion(nu, f, dx, dy, scheme="2nd"):
    """
    diffuse f with diffusion rate nu
    """

    if scheme == "2nd":
        # 2nd order accurate, 2nd order derivatives
        f_xx = (f[3:-1, 2:-2] - 2. * f[2:-2, 2:-2] + f[1:-3, 2:-2]) / dx**2
        f_yy = (f[2:-2, 3:-1] - 2. * f[2:-2, 2:-2] + f[2:-2, 1:-3]) / dy**2

    elif scheme == "4th":
        # 4th order accurate
        f_xx = (- f[4:, 2:-2] + 16. * f[3:-1, 2:-2] - 30. * f[2:-2, 2:-2] + 16. * f[1:-3, 2:-2] - f[:-4, 2:-2]) / (12. * dx**2)
        f_yy = (- f[2:-2, 4:] + 16. * f[2:-2, 3:-1] - 30. * f[2:-2, 2:-2] + 16. * f[2:-2, 1:-3] - f[2:-2, :-4]) / (12. * dy**2)

    # diffusion
    lap = f_xx + f_yy
    diff = nu * lap

    return diff


def divergence(u, v, dx, dy):
    """
    divergence of velocity field
    evaluate at cell center
    """
    # map to edge
    u_mapped = 1. / 2. * (u[:, 1:] + u[:, :-1])   # shifted along y-axis
    v_mapped = 1. / 2. * (v[1:, :] + v[:-1, :])   # shifted along x-axis

    # 1st order derivatives at cell center
    u_x = (u_mapped[1:, :] - u_mapped[:-1, :]) / dx
    v_y = (v_mapped[:, 1:] - v_mapped[:, :-1]) / dy

    # divergence at cell center
    div = u_x + v_y
    return div


def gradient(p, dx, dy):
    """
    pressure gradient
    evaluate at vertex
    """
    # map to edge
    p_mapped_u = 1. / 2. * (p[:, 1:] + p[:, :-1])   # shifted along y-axis, used for u
    p_mapped_v = 1. / 2. * (p[1:, :] + p[:-1, :])   # shifted along x-axis, used for v

    # pressure gradient at vertex
    p_x = (p_mapped_u[1:, :] - p_mapped_u[:-1, :]) / dx
    p_y = (p_mapped_v[:, 1:] - p_mapped_v[:, :-1]) / dy

    # drop boundary points
    p_x = p_x[1:-1, 1:-1]
    p_y = p_y[1:-1, 1:-1]
    return p_x, p_y


def vorticity(u, v, dx, dy):
    """
    vorticity of velocity field
    evaluate at cell center
    """
    # map to edge
    u_mapped = 1. / 2. * (u[1:, :] + u[:-1, :])   # shifted along x-axis
    v_mapped = 1. / 2. * (v[:, 1:] + v[:, :-1])   # shifted along y-axis

    # 1st order derivatives at cell center
    u_y = (u_mapped[:, 1:] - u_mapped[:, :-1]) / dy
    v_x = (v_mapped[1:, :] - v_mapped[:-1, :]) / dx

    # vorticity at cell center
    omega = v_x - u_y
    return omega


def interpolate(p, u, v):
    """
    interpolate pressure to velocity nodes
    """
    # 4 node to 1 node interpolation
    p_interp = 1. / 4. * (p[:-1, 1:] + p[1:, 1:] + p[:-1, :-1] + p[1:, :-1])

    # shape check
    print(f"p.shape: {p.shape}, u.shape: {u.shape}, v.shape: {v.shape}, p_interp.shape: {p_interp.shape}")

    return p_interp
