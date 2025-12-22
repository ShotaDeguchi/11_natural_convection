"""
********************************************************************************
natural convection in a rectangular cavity
********************************************************************************
"""

import time
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import skimage

import operators_3D

################################################################################

parser = argparse.ArgumentParser()
parser.add_argument("--rho", type=float, default=1e3, help="density")
parser.add_argument("--mu", type=float, default=1e-3, help="dynamic viscosity")
# parser.add_argument("--nu", type=float, default=1e-6, help="kinematic viscosity")   # computed
parser.add_argument("--lam", type=float, default=.6, help="thermal conductivity")
parser.add_argument("--cap", type=float, default=4.18e3, help="specific heat capacity")
# parser.add_argument("--kap", type=float, default=1.44e-7, help="thermal diffusivity")   # computed
parser.add_argument("--beta", type=float, default=2.1e-4, help="thermal expansion coefficient")
parser.add_argument("--grav", type=float, default=9.81, help="gravitational acceleration")
# parser.add_argument("-p", "--Pr", type=float, default=7., help="Prandtl number")
# parser.add_argument("-r", "--Ra", type=float, default=1e4, help="Rayleigh number")
args = parser.parse_args()

################################################################################

def main():
    # seed
    np.random.seed(42)

    # domain
    Lx, Ly, Lz = 2., 2., 1.
    h_res = 2e-2
    dx, dy, dz = h_res, h_res, h_res

    # arakawa-b grid
    x_vel, y_vel, z_vel = np.arange(0.-dx, Lx+2.*dx, dx), np.arange(0.-dy, Ly+2.*dy, dy), np.arange(0.-dz, Lz+2.*dz, dz)
    Nx_vel, Ny_vel, Nz_vel = len(x_vel), len(y_vel), len(z_vel)
    x_vel, y_vel, z_vel = np.meshgrid(x_vel, y_vel, z_vel, indexing="ij")

    x_prs, y_prs, z_prs = np.arange(0.-1./2.*dx, Lx+3./2.*dx, dx), np.arange(0.-1./2.*dy, Ly+3./2.*dy, dy), np.arange(0.-1./2.*dz, Lz+3./2.*dz, dz)
    Nx_prs, Ny_prs, Nz_prs = len(x_prs), len(y_prs), len(z_prs)
    x_prs, y_prs, z_prs = np.meshgrid(x_prs, y_prs, z_prs, indexing="ij")

    # boundary temperature
    h0, h1 = 20., 100.
    h0, h1 = h0 + 273.15, h1 + 273.15
    theta = h1 - h0

    # params of the fluid
    rho = 1e3   # density [kg / m^3]
    mu = 1e-3   # dynamic viscosity [Pa \cdot s]
    nu = mu / rho   # kinematic viscosity [m^2 / s]
    lam = .6   # thermal conductivity [W / (m \cdot K)]
    cap = 4.18e3   # specific heat capacity [J / (kg \cdot K)]
    kap = lam / (rho * cap)   # thermal diffusivity [m^2 / s]
    beta = 2.1e-4   # thermal expansion coefficient [1 / K]
    grav = 9.81   # gravitational acceleration [m / s^2]
    # grav = np.array([grav, grav, grav])   # vector
    Pr = nu / kap
    Gr = (grav * beta * Lz**3 * theta) / nu**2
    Ra = Pr * Gr

    # boundary condition
    # bc = "cold"  # "natural" / "periodic"
    bc = "natural"  # "natural" / "periodic"
    # bc = "periodic"  # "natural" / "periodic"

    # path
    path_res = Path(f"res_3D_Pr{Pr:.3e}_Ra{Ra:.3e}_{bc}")
    # path_res = Path(f"res_3D_Pr{Pr:.3e}_Ra{Ra:.3e}_{bc}_test")
    path_fig = path_res / "fig"
    path_fig.mkdir(parents=True, exist_ok=True)
    path_npz = path_res / "npz"
    path_npz.mkdir(parents=True, exist_ok=True)

    # variables
    u = np.zeros((Nx_vel, Ny_vel, Nz_vel)) + 0. * np.random.normal(size=(Nx_vel, Ny_vel, Nz_vel))  # velocity in x-direction
    v = np.zeros((Nx_vel, Ny_vel, Nz_vel)) + 0. * np.random.normal(size=(Nx_vel, Ny_vel, Nz_vel))  # velocity in y-direction
    w = np.zeros((Nx_vel, Ny_vel, Nz_vel)) + 0. * np.random.normal(size=(Nx_vel, Ny_vel, Nz_vel))  # velocity in z-direction
    p = np.zeros((Nx_prs, Ny_prs, Nz_prs)) + 0. * np.random.normal(size=(Nx_prs, Ny_prs, Nz_prs))  # pressure
    b = np.zeros((Nx_prs, Ny_prs, Nz_prs)) + 0. * np.random.normal(size=(Nx_prs, Ny_prs, Nz_prs))  # source for pressure poisson eq
    h = h0 * np.ones((Nx_vel, Ny_vel, Nz_vel)) + 1. * np.random.normal(size=(Nx_vel, Ny_vel, Nz_vel))  # temperature (no need for mapping)

    h = h0 + (h1 - h0) * (1. - z_vel / Lz) + 1. * np.random.normal(size=(Nx_vel, Ny_vel, Nz_vel))  # h(y=Ly)=h0 (top), h(y=0)=h1 (bottom)

    # time
    dim = 3.
    Ux = 1.
    dt1 = 1. * dx**1 / (Ux * dim)
    dt2 = .5 * dx**2 / (nu * dim)
    dt3 = .5 * dx**2 / (kap * dim)
    dt = min(dt1, dt2, dt3)
    print(f"dt1: {dt1:.3e}, dt2: {dt2:.3e}, dt3: {dt3:.3e}")
    dt *= .4

    dt = dx / 20.
    T = 60. * 1.

    maxiter_vel = int(T / dt)
    maxiter_ppe = int(1e4)
    tol_vel = 1e-6
    tol_ppe = 1e-6

    # main
    t = 0.
    it_vel = 0
    t -= dt
    it_vel -= 1
    res_vel = np.inf
    while it_vel < maxiter_vel:
        # update
        t += dt
        it_vel += 1

        # previous velocity
        u_old = np.copy(u)
        v_old = np.copy(v)
        w_old = np.copy(w)

        # intermediate velocity
        u_hat = np.copy(u)
        v_hat = np.copy(v)
        w_hat = np.copy(w)

        # advection
        u_advc = operators_3D.advection(u_old, v_old, w_old, u_old, dx, dy, dz, dt, scheme="KK")
        v_advc = operators_3D.advection(u_old, v_old, w_old, v_old, dx, dy, dz, dt, scheme="KK")
        w_advc = operators_3D.advection(u_old, v_old, w_old, w_old, dx, dy, dz, dt, scheme="KK")

        # diffusion
        u_diff = operators_3D.diffusion(nu, u_old, dx, dy, dz)
        v_diff = operators_3D.diffusion(nu, v_old, dx, dy, dz)
        w_diff = operators_3D.diffusion(nu, w_old, dx, dy, dz)

        # buoyancy
        buoyancy = (1. - beta * (h - h0)) * (- grav)
        # buoyancy = (1. - beta * (h - h1)) * grav
        # buoyancy = beta * (h - h0) * grav
        # print(f"buoyancy.min(): {buoyancy.min():.6f}")
        # print(f"buoyancy.max(): {buoyancy.max():.6f}")

        # intermediate velocity
        u_hat[2:-2, 2:-2, 2:-2] = u_old[2:-2, 2:-2, 2:-2] + dt * (- u_advc + u_diff)
        v_hat[2:-2, 2:-2, 2:-2] = v_old[2:-2, 2:-2, 2:-2] + dt * (- v_advc + v_diff)
        w_hat[2:-2, 2:-2, 2:-2] = w_old[2:-2, 2:-2, 2:-2] + dt * (- w_advc + w_diff + buoyancy[2:-2, 2:-2, 2:-2])

        # pressure poisson eq
        div_hat = operators_3D.velocity_divergence(u_hat, v_hat, w_hat, dx, dy, dz)
        b = rho / dt * div_hat
        for it_ppe in range(maxiter_ppe+1):
            p_old = p.copy()
            p[1:-1, 1:-1, 1:-1] = 1. / (2. * (dy**2 * dz**2 + dz**2 * dx**2 + dx**2 * dy**2)) \
                            * (
                                - b * dx**2 * dy**2 * dz**2 \
                                + (p_old[2:, 1:-1, 1:-1] + p_old[:-2, 1:-1, 1:-1]) * dy**2 * dz**2 \
                                + (p_old[1:-1, 2:, 1:-1] + p_old[1:-1, :-2, 1:-1]) * dz**2 * dx**2 \
                                + (p_old[1:-1, 1:-1, 2:] + p_old[1:-1, 1:-1, :-2]) * dx**2 * dy**2
                            )

            if bc == "natural" or bc == "cold":
                # Neumann boundary condition
                p[0,  :, :] = p[1,  :, :]   # x = xmin plane
                p[-1, :, :] = p[-2, :, :]   # x = xmax plane
                p[:,  0, :] = p[:,  1, :]   # y = ymin plane
                p[:, -1, :] = p[:, -2, :]   # y = ymax plane
                p[:, :,  0] = p[:, :,  1]   # z = zmin plane
                p[:, :, -1] = p[:, :, -2]   # z = zmax plane
                # p[1, 1, 1] = 0.   # (x, y, z) = (xmin, ymin, zmin) corner

            elif bc == "periodic":
                # periodic boundary conditions
                p[0,  :, :] = p[-2, :, :]   # West = East interior
                p[-1, :, :] = p[1,  :, :]   # East = West interior
                p[:,  0, :] = p[:, -2, :]   # South = North interior
                p[:, -1, :] = p[:,  1, :]   # North = South interior
                p[:, :,  0] = p[:, :,  1]   # z = zmin plane
                p[:, :, -1] = p[:, :, -2]   # z = zmax plane

            # convergence?
            p_flat = p.flatten()
            p_old_flat = p_old.flatten()
            res_ppe = np.linalg.norm(p_flat - p_old_flat, 2) / np.linalg.norm(p_old_flat, 2)
            if it_ppe % int(maxiter_ppe / 5) == 0:
                print(f"[PPE] it_ppe: {it_ppe:06d} / {maxiter_ppe:06d}, res_ppe: {res_ppe:.6e} / {tol_ppe:.6e}")
            if res_ppe < tol_ppe:
                print(f"[PPE] it_ppe: {it_ppe:06d} / {maxiter_ppe:06d}, res_ppe: {res_ppe:.6e} / {tol_ppe:.6e}")
                print(f"[PPE] converged")
                break

        p_x, p_y, p_z = operators_3D.pressure_gradient(p, dx, dy, dz)
        u[2:-2, 2:-2, 2:-2] = u_hat[2:-2, 2:-2, 2:-2] + dt * (- p_x / rho)
        v[2:-2, 2:-2, 2:-2] = v_hat[2:-2, 2:-2, 2:-2] + dt * (- p_y / rho)
        w[2:-2, 2:-2, 2:-2] = w_hat[2:-2, 2:-2, 2:-2] + dt * (- p_z / rho)

        if bc == "natural" or bc == "cold":
            # no-slip boundary condition
            u[:2,  :, :], v[:2,  :, :], w[:2,  :, :] = 0., 0., 0.   # x = xmin plane
            u[-2:, :, :], v[-2:, :, :], w[-2:, :, :] = 0., 0., 0.   # x = xmax plane
            u[:,  :2, :], v[:,  :2, :], w[:,  :2, :] = 0., 0., 0.   # y = ymin plane
            u[:, -2:, :], v[:, -2:, :], w[:, -2:, :] = 0., 0., 0.   # y = ymax plane
            u[:, :,  :2], v[:, :,  :2], w[:, :,  :2] = 0., 0., 0.   # z = zmin plane
            u[:, :, -2:], v[:, :, -2:], w[:, :, -2:] = 0., 0., 0.   # z = zmax plane

        elif bc == "periodic":
            # periodic boundary condition
            u[ 0, :, :], v[ 0, :, :], w[ 0, :, :] = u[-4, :, :], v[-4, :, :], w[-4, :, :]   # x = xmin plane
            u[ 1, :, :], v[ 1, :, :], w[ 1, :, :] = u[-3, :, :], v[-3, :, :], w[-3, :, :]   # x = xmin plane
            u[-1, :, :], v[-1, :, :], w[-1, :, :] = u[ 3, :, :], v[ 3, :, :], w[ 3, :, :]   # x = xmax plane
            u[-2, :, :], v[-2, :, :], w[-2, :, :] = u[ 2, :, :], v[ 2, :, :], w[ 2, :, :]   # x = xmax plane

            u[:,  0, :], v[:,  0, :], w[:,  0, :] = u[:, -4, :], v[:, -4, :], w[:, -4, :]   # y = ymin plane
            u[:,  1, :], v[:,  1, :], w[:,  1, :] = u[:, -3, :], v[:, -3, :], w[:, -3, :]   # y = ymin plane
            u[:, -1, :], v[:, -1, :], w[:, -1, :] = u[:,  3, :], v[:,  3, :], w[:,  3, :]   # y = ymax plane
            u[:, -2, :], v[:, -2, :], w[:, -2, :] = u[:,  2, :], v[:,  2, :], w[:,  2, :]   # y = ymax plane

            u[:, :,  :2], v[:, :,  :2], w[:, :,  :2] = 0., 0., 0.   # z = zmin plane
            u[:, :, -2:], v[:, :, -2:], w[:, :, -2:] = 0., 0., 0.   # z = zmax plane

        # convergence?
        u_flat = u.flatten()
        v_flat = v.flatten()
        w_flat = w.flatten()
        u_old_flat = u_old.flatten()
        v_old_flat = v_old.flatten()
        w_old_flat = w_old.flatten()
        res_u = np.linalg.norm(u_flat - u_old_flat, 2) / np.linalg.norm(u_old_flat, 2)
        res_v = np.linalg.norm(v_flat - v_old_flat, 2) / np.linalg.norm(v_old_flat, 2)
        res_w = np.linalg.norm(w_flat - w_old_flat, 2) / np.linalg.norm(w_old_flat, 2)
        res_vel = max(res_u, res_v, res_w)
        # if res_vel < tol_vel:
        #     print(f"[MAIN] converged")
        #     break
        print(f"\n****************************************************************")
        print(f"[MAIN] t: {t:.3f} / {T:.3f}")
        print(f"[MAIN] it_vel: {it_vel:06d} / {maxiter_vel:06d}")
        print(f"[MAIN] dx: {dx:.3e}, dt: {dt:.3e}")
        print(f"[MAIN] res_vel: {res_vel:.6e} / {tol_vel:.6e}")

        C = np.max(np.abs(u) * dt / dx + np.abs(v) * dt / dy + np.abs(w) * dt / dz)
        D = np.max(nu * dt  / dx**2 + nu * dt  / dy**2 + nu * dt  / dz**2)
        K = np.max(kap * dt / dx**2 + kap * dt / dy**2 + kap * dt / dz**2)
        print(f"[MAIN] Courant number    : {C:.6f} < 1.0")
        print(f"[MAIN] diffusion number  : {D:.6f} < 0.5")
        print(f"[MAIN] diffusivity number: {K:.6f} < 0.5")

        vel_norm = np.sqrt(u**2 + v**2 + w**2)
        Re = np.max(vel_norm * Lx / nu)
        print(f"[MAIN] Reynolds number: {Re:.3e}")
        print(f"[MAIN] Prandtl number : {Pr:.3e}")
        print(f"[MAIN] Grashof number : {Gr:.3e}")
        print(f"[MAIN] Rayleigh number: {Ra:.3e}")
        print(f"****************************************************************")

        # print(f"[MAIN] dt: {dt:.3e}")
        # dt1 = 1. * dx**1 / (np.max(vel_norm) * dim)
        # dt2 = .5 * dx**2 / (nu * dim)
        # dt3 = .5 * dx**2 / (kap * dim)
        # dt = min(dt1, dt2, dt3)
        # dt *= .4
        # print(f"[MAIN] dt1: {dt1:.3e}")
        # print(f"[MAIN] dt2: {dt2:.3e}")
        # print(f"[MAIN] dt3: {dt3:.3e}")
        # print(f"[MAIN] dt: {dt:.3e}")

        ########################################################################
        # temperature
        h_old = np.copy(h)
        h_advc = operators_3D.advection(u, v, w, h, dx, dy, dz, dt, scheme="KK")
        h_diff = operators_3D.diffusion(kap, h, dx, dy, dz)
        h[2:-2, 2:-2, 2:-2] = h_old[2:-2, 2:-2, 2:-2] + dt * (- h_advc + h_diff)

        if bc == "natural":
            # Neumann boundary condition
            h[:2,  :, :] = h[2,  :, :]   # x = xmin plane
            h[-2:, :, :] = h[-3, :, :]   # x = xmax plane
            h[:,  :2, :] = h[:,  2:3, :]   # y = ymin plane
            h[:, -2:, :] = h[:, -3:-2, :]   # y = ymax plane
            h[:, :,  :2] = h1   # z = zmin plane
            h[:, :, -2:] = h0   # z = zmax plane

        elif bc == "cold":
            # Dirichlet boundary condition
            h[:2,  :, :] = h0   # x = xmin plane
            h[-2:, :, :] = h0   # x = xmax plane
            h[:,  :2, :] = h0   # y = ymin plane
            h[:, -2:, :] = h0   # y = ymax plane
            h[:, :,  :2] = h1   # z = zmin plane
            h[:, :, -2:] = h0   # z = zmax plane

        elif bc == "periodic":
            # periodic boundary condition
            h[ 0, :, :]  = h[-4, :, :]; h[ 1, :, :] = h[-3, :, :]   # x = xmin plane
            h[-1, :, :]  = h[ 3, :, :]; h[-2, :, :] = h[2,  :, :]   # x = xmax plane
            h[:,  0, :]  = h[:, -4, :]; h[:,  1, :] = h[:, -3, :]   # y = ymin plane
            h[:, -1, :]  = h[:,  3, :]; h[:, -2, :] = h[:,  2, :]   # y = ymax plane
            h[:, :,  :2] = h1   # z = zmin plane
            h[:, :, -2:] = h0   # z = zmax plane

        # # print to check periodicity
        # print(f"\nh[ 0, Ny_vel//2, Nz_vel//2]: {h[0, Ny_vel//2, Nz_vel//2]}")
        # print(f"h[ 1, Ny_vel//2, Nz_vel//2]: {h[1, Ny_vel//2, Nz_vel//2]}")
        # print(f"h[ 2, Ny_vel//2, Nz_vel//2]: {h[2, Ny_vel//2, Nz_vel//2]}")
        # print(f"h[ 3, Ny_vel//2, Nz_vel//2]: {h[3, Ny_vel//2, Nz_vel//2]}")
        # print(f"h[-1, Ny_vel//2, Nz_vel//2]: {h[-1, Ny_vel//2, Nz_vel//2]}")
        # print(f"h[-2, Ny_vel//2, Nz_vel//2]: {h[-2, Ny_vel//2, Nz_vel//2]}")
        # print(f"h[-3, Ny_vel//2, Nz_vel//2]: {h[-3, Ny_vel//2, Nz_vel//2]}")
        # print(f"h[-4, Ny_vel//2, Nz_vel//2]: {h[-4, Ny_vel//2, Nz_vel//2]}")

        # print(f"\nu[ 0, Ny_vel//2, Nz_vel//2]: {u[0, Ny_vel//2, Nz_vel//2]}")
        # print(f"u[ 1, Ny_vel//2, Nz_vel//2]: {u[1, Ny_vel//2, Nz_vel//2]}")
        # print(f"u[ 2, Ny_vel//2, Nz_vel//2]: {u[2, Ny_vel//2, Nz_vel//2]}")
        # print(f"u[ 3, Ny_vel//2, Nz_vel//2]: {u[3, Ny_vel//2, Nz_vel//2]}")
        # print(f"u[-1, Ny_vel//2, Nz_vel//2]: {u[-1, Ny_vel//2, Nz_vel//2]}")
        # print(f"u[-2, Ny_vel//2, Nz_vel//2]: {u[-2, Ny_vel//2, Nz_vel//2]}")
        # print(f"u[-3, Ny_vel//2, Nz_vel//2]: {u[-3, Ny_vel//2, Nz_vel//2]}")
        # print(f"u[-4, Ny_vel//2, Nz_vel//2]: {u[-4, Ny_vel//2, Nz_vel//2]}")

        # print(f"\nv[ 0, Ny_vel//2, Nz_vel//2]: {v[0, Ny_vel//2, Nz_vel//2]}")
        # print(f"v[ 1, Ny_vel//2, Nz_vel//2]: {v[1, Ny_vel//2, Nz_vel//2]}")
        # print(f"v[ 2, Ny_vel//2, Nz_vel//2]: {v[2, Ny_vel//2, Nz_vel//2]}")
        # print(f"v[ 3, Ny_vel//2, Nz_vel//2]: {v[3, Ny_vel//2, Nz_vel//2]}")
        # print(f"v[-1, Ny_vel//2, Nz_vel//2]: {v[-1, Ny_vel//2, Nz_vel//2]}")
        # print(f"v[-2, Ny_vel//2, Nz_vel//2]: {v[-2, Ny_vel//2, Nz_vel//2]}")
        # print(f"v[-3, Ny_vel//2, Nz_vel//2]: {v[-3, Ny_vel//2, Nz_vel//2]}")
        # print(f"v[-4, Ny_vel//2, Nz_vel//2]: {v[-4, Ny_vel//2, Nz_vel//2]}")

        # print(f"\nw[ 0, Ny_vel//2, Nz_vel//2]: {w[0, Ny_vel//2, Nz_vel//2]}")
        # print(f"w[ 1, Ny_vel//2, Nz_vel//2]: {w[1, Ny_vel//2, Nz_vel//2]}")
        # print(f"w[ 2, Ny_vel//2, Nz_vel//2]: {w[2, Ny_vel//2, Nz_vel//2]}")
        # print(f"w[ 3, Ny_vel//2, Nz_vel//2]: {w[3, Ny_vel//2, Nz_vel//2]}")
        # print(f"w[-1, Ny_vel//2, Nz_vel//2]: {w[-1, Ny_vel//2, Nz_vel//2]}")
        # print(f"w[-2, Ny_vel//2, Nz_vel//2]: {w[-2, Ny_vel//2, Nz_vel//2]}")
        # print(f"w[-3, Ny_vel//2, Nz_vel//2]: {w[-3, Ny_vel//2, Nz_vel//2]}")
        # print(f"w[-4, Ny_vel//2, Nz_vel//2]: {w[-4, Ny_vel//2, Nz_vel//2]}")

        # print(f"\np[ 0, Ny_vel//2, Nz_vel//2]: {p[0, Ny_vel//2, Nz_vel//2]}")
        # print(f"p[ 1, Ny_vel//2, Nz_vel//2]: {p[1, Ny_vel//2, Nz_vel//2]}")
        # print(f"p[ 2, Ny_vel//2, Nz_vel//2]: {p[2, Ny_vel//2, Nz_vel//2]}")
        # print(f"p[ 3, Ny_vel//2, Nz_vel//2]: {p[3, Ny_vel//2, Nz_vel//2]}")
        # print(f"p[-1, Ny_vel//2, Nz_vel//2]: {p[-1, Ny_vel//2, Nz_vel//2]}")
        # print(f"p[-2, Ny_vel//2, Nz_vel//2]: {p[-2, Ny_vel//2, Nz_vel//2]}")
        # print(f"p[-3, Ny_vel//2, Nz_vel//2]: {p[-3, Ny_vel//2, Nz_vel//2]}")
        # print(f"p[-4, Ny_vel//2, Nz_vel//2]: {p[-4, Ny_vel//2, Nz_vel//2]}")

        ########################################################################

        plot_every = 1.   # plot every x seconds
        if it_vel % int(plot_every / dt) == 0:

        # if it_vel % 400 == 0:
        # if it_vel % 800 == 0:
        # if it_vel % 1000 == 0:
            # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

            # cf = ax.scatter(
            #     x_vel[2:-2, 2:-2, 2:-2], y_vel[2:-2, 2:-2, 2:-2], z_vel[2:-2, 2:-2, 2:-2],
            #     c=h[2:-2, 2:-2, 2:-2],
            #     vmin=h0, vmax=h1, cmap="magma", marker=".", alpha=.5
            # )

            # # Add cube wireframe
            # cube_vertices = [
            #     [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # bottom face
            #     [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]   # top face
            # ]
            # cube_edges = [
            #     [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face edges
            #     [4, 5], [5, 6], [6, 7], [7, 4],  # top face edges
            #     [0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
            # ]
            # for edge in cube_edges:
            #     points = [cube_vertices[edge[0]], cube_vertices[edge[1]]]
            #     ax.plot3D(*zip(*points), "k-", alpha=.3, linewidth=.5)

            # ax.set(
            #     xticks=np.linspace(0., Lx, 3),
            #     yticks=np.linspace(0., Ly, 3),
            #     zticks=np.linspace(0., Lz, 3),
            #     xlim=(0., Lx),
            #     ylim=(0., Ly),
            #     zlim=(0., Lz),
            #     # xlabel=r"$x$",
            #     # ylabel=r"$y$",
            #     # zlabel=r"$z$",
            #     title=rf"$t = {t:.3f} \ [\text{{s}}]$",
            #     aspect="equal",
            # )
            # fig.tight_layout()
            # fig.savefig(path_res / f"res.png")
            # fig.savefig(path_res / f"res_it{it_vel:06d}.png")
            # plt.close(fig)


            # visualize
            mask = (x_vel < Lx/2.) | (y_vel > Ly/2.)
            h_masked = np.where(mask, h, np.nan)

            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

            iso_vals = np.linspace(np.round(h0, 2), np.round(h1, 2), 5)
            iso_vals = np.linspace(np.round(h0+10., 2), np.round(h1-10., 2), 5)
            colors = plt.cm.turbo(np.linspace(0., 1., len(iso_vals)))
            alphas = np.linspace(.8, .9, len(iso_vals))
            # alphas = np.linspace(.1, .2, len(iso_vals))
            # alphas = alphas[::-1]  # reverse order for better visibility
            norm = mcolors.Normalize(vmin=min(iso_vals), vmax=max(iso_vals))
            spacing = (dx, dy, dz)
            for iso_val, color, alpha in zip(iso_vals, colors, alphas):
                try:
                    verts, faces, _, _ = skimage.measure.marching_cubes(
                        h_masked[1:-1, 1:-1, 1:-1], level=iso_val, spacing=spacing,
                        # h[1:-1, 1:-1, 1:-1], level=iso_val, spacing=spacing,
                    )
                    ax.plot_trisurf(
                        verts[:, 0], verts[:, 1], verts[:, 2], triangles=faces,
                        color=color, alpha=alpha, shade=True,
                    )
                except:
                    continue
            handles = [
                plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=color, alpha=alpha, label=rf"$T = {iso_val} \text{{ [K]}}$")
                for color, alpha, iso_val in zip(colors, alphas, iso_vals)
            ]
            ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(.95, 1.1))

            iso_vals = np.linspace(h0, h1, 5)
            iso_vals = np.linspace(h0+10., h1-10., 5)
            colors = plt.cm.turbo(np.linspace(0., 1., len(iso_vals)))
            # alphas = np.linspace(.8, .9, len(iso_vals))
            alphas = np.linspace(.1, .2, len(iso_vals))
            # alphas = alphas[::-1]  # reverse order for better visibility
            norm = mcolors.Normalize(vmin=min(iso_vals), vmax=max(iso_vals))
            spacing = (dx, dy, dz)
            for iso_val, color, alpha in zip(iso_vals, colors, alphas):
                try:
                    verts, faces, _, _ = skimage.measure.marching_cubes(
                        # h_masked[1:-1, 1:-1, 1:-1], level=iso_val, spacing=spacing,
                        h[1:-1, 1:-1, 1:-1], level=iso_val, spacing=spacing,
                    )
                    ax.plot_trisurf(
                        verts[:, 0], verts[:, 1], verts[:, 2], triangles=faces,
                        color=color, alpha=alpha, shade=True,
                    )
                except:
                    continue
            # handles = [
            #     plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=color, alpha=alpha, label=rf"$T = {iso_val} \text{{ [K]}}$")
            #     for color, alpha, iso_val in zip(colors, alphas, iso_vals)
            # ]
            # ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(.95, 1.1))

            cube_vertices = [
                [0, 0, 0],  [Lx, 0, 0],  [Lx, Ly, 0],  [0, Ly, 0],   # bottom face
                [0, 0, Lz], [Lx, 0, Lz], [Lx, Ly, Lz], [0, Ly, Lz]   # top face
            ]
            cube_edges = [
                [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face edges
                [4, 5], [5, 6], [6, 7], [7, 4],  # top face edges
                [0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
            ]
            for edge in cube_edges:
                points = [cube_vertices[edge[0]], cube_vertices[edge[1]]]
                ax.plot3D(*zip(*points), "k-", alpha=.3, zorder=10)
            ax.set(
                xticks=np.linspace(0., Lx, 2),
                yticks=np.linspace(0., Ly, 2),
                zticks=np.linspace(0., Lz, 2),
                xlim=(0., Lx),
                ylim=(0., Ly),
                zlim=(0., Lz),
                # xlabel=rf"$x \text{{ [m]}}$",
                # ylabel=rf"$y \text{{ [m]}}$",
                # zlabel=rf"$z \text{{ [m]}}$",
                # title=rf"$t={t:.3f} / {T:.3f} \text{{ [s]}}$",
                title=rf"$t={t:.3f} \text{{ [s]}}$",
                aspect="equal",
            )
            fig.tight_layout()
            fig.savefig(path_fig / f"fig.png")
            fig.savefig(path_fig / f"fig_t{t:.3f}.png")
            # fig.savefig(path_fig / f"fig_it{it_vel:06d}.png")
            plt.close(fig)

            # save data in numpy format
            np.savez(
                path_npz / f"res_it{it_vel:06d}.npz",
                u=u, v=v, w=w, p=p, h=h,
                x_vel=x_vel, y_vel=y_vel, z_vel=z_vel,
                x_prs=x_prs, y_prs=y_prs, z_prs=z_prs,
                Lx=Lx, Ly=Ly, Lz=Lz,
                dx=dx, dy=dy, dz=dz,
                t=t
            )

################################################################################

def plot_setting():
    plt.style.use("default")
    plt.style.use("seaborn-v0_8-deep")
    plt.style.use("seaborn-v0_8-talk")   # paper / notebook / talk / poster
    # plt.style.use("classic")
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["figure.figsize"] = (7, 5)
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["axes.grid"] = True
    # plt.rcParams["axes.axisbelow"] = True   # background grid
    plt.rcParams["grid.alpha"] = .3
    plt.rcParams["legend.framealpha"] = .8
    plt.rcParams["legend.facecolor"] = "w"
    plt.rcParams["savefig.dpi"] = 300

################################################################################

if __name__ == "__main__":
    plot_setting()
    main()
