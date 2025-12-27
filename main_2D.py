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

import operators_2D
import solvers

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
# parser.add_argument("-p", "--Prf", type=float, default=7., help="Prandtl numberf")
# parser.add_argument("-rf", "--Ra", type=float, default=1e4, help="Rayleigh numberf")
args = parser.parse_args()

################################################################################

def main():
    # seed
    np.random.seed(42)

    # domain
    Lx, Ly = 8., 1.
    # Lx, Ly = 1., 2.
    h_res = 5e-3
    dx, dy = h_res, h_res

    # arakawa-b grid
    x_vel, y_vel = np.arange(0.-dx, Lx+2.*dx, dx), np.arange(0.-dy, Ly+2.*dy, dy)
    Nx_vel, Ny_vel = len(x_vel), len(y_vel)
    x_vel, y_vel = np.meshgrid(x_vel, y_vel, indexing="ij")

    x_prs, y_prs = np.arange(0.-1./2.*dx, Lx+3./2.*dx, dx), np.arange(0.-1./2.*dy, Ly+3./2.*dy, dy)
    Nx_prs, Ny_prs = len(x_prs), len(y_prs)
    x_prs, y_prs = np.meshgrid(x_prs, y_prs, indexing="ij")

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
    Pr = nu / kap
    Gr = (grav * beta * Ly**3 * theta) / nu**2
    Ra = Pr * Gr

    # boundary condition
    bc = "natural"  # "natural" / "periodic"
    # bc = "cold"  # "natural" / "periodic"
    # bc = "periodic"  # "natural" / "periodic"

    # path
    # path_res = Path(f"res_2D_Pr{Pr:.3e}_Ra{Ra:.3e}_{bc}")
    path_res = Path(f"res_2D_test")
    path_res.mkdir(exist_ok=True)

    # variables
    u = np.zeros((Nx_vel, Ny_vel)) + 0. * np.random.normal(size=(Nx_vel, Ny_vel))  # velocity in x-direction
    v = np.zeros((Nx_vel, Ny_vel)) + 0. * np.random.normal(size=(Nx_vel, Ny_vel))  # velocity in y-direction
    p = np.zeros((Nx_prs, Ny_prs)) + 0. * np.random.normal(size=(Nx_prs, Ny_prs))  # pressure
    b = np.zeros((Nx_prs, Ny_prs)) + 0. * np.random.normal(size=(Nx_prs, Ny_prs))  # source for pressure poisson eq
    h = h0 * np.ones((Nx_vel, Ny_vel))
    # h = h0 + (h1 - h0) * (1. - x_vel / Lx)
    # h = h0 + (h1 - h0) * (1. - y_vel / Ly)
    h[:Nx_vel//2, :] = h1
    # h[:, :Ny_vel//2] = h1

    # # sine-wave shaped interface
    # amp = Ly / 20.     # amplitude of wave
    # wav = 2. * np.pi / Lx  # one full wave across the x-domain
    # interface = Ly / 2. + amp * np.sin(wav * x_vel)
    # h = np.where(y_vel < interface, h1, h0)

    # # straight interface (bottom left to top right)
    # interface = (Ly / Lx) * x_vel
    # h = np.where(y_vel < interface, h1, h0)

    # # layered structure
    # n_layers = 8
    # layer_height = Ly / n_layers
    # for i in range(n_layers):
    #     if i % 2 == 0:
    #         h[:, int(i * layer_height / dy):int((i+1) * layer_height / dy)] = h0
    #         # u[:, int(i * layer_height / dy):int((i+1) * layer_height / dy)] = -1.
    #     else:
    #         h[:, int(i * layer_height / dy):int((i+1) * layer_height / dy)] = h1
    #         # u[:, int(i * layer_height / dy):int((i+1) * layer_height / dy)] = 1.

    # # gaussian blob
    # x0, y0, r0 = Lx / 2., Ly / 2., min(Lx, Ly) / 4.
    # dist = np.sqrt((x_vel - x0)**2 + (y_vel - y0)**2)
    # blob = np.exp(-(dist**2) / (2 * (r0 / 2.)**2))
    # h += (h1 - h0) * blob

    # # rotational initial velocity
    # omega = 1.
    # u = omega * (y_vel - y0)
    # v = -omega * (x_vel - x0)

    # # decay the velocity toward the boundary
    # rot_mask = np.exp(-(dist**2) / (2 * r0**2))
    # u = u * rot_mask
    # v = v * rot_mask

    # h = h[:,::-1]  # flip y-axis
    # h += np.random.normal(size=(Nx_vel, Ny_vel))   # add perturbation

    # time
    n_dims = 2.
    Ux = 1.
    dt1 = 1. * dx**1 / (Ux * n_dims)
    dt2 = .5 * dx**2 / (nu * n_dims)
    dt3 = .5 * dx**2 / (kap * n_dims)
    dt = min(dt1, dt2, dt3)
    print(f"dt1: {dt1:.3e}, dt2: {dt2:.3e}, dt3: {dt3:.3e}")
    dt *= .4

    T = 60. * 1.
    dump_out_interval = .5  # plot every ??? seconds

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

        # intermediate velocity
        u_hat = np.copy(u)
        v_hat = np.copy(v)

        # advection
        u_advc = operators_2D.advection(u_old, v_old, u_old, dx, dy, dt, scheme="KK")
        v_advc = operators_2D.advection(u_old, v_old, v_old, dx, dy, dt, scheme="KK")

        # diffusion
        u_diff = operators_2D.diffusion(nu, u_old, dx, dy)
        v_diff = operators_2D.diffusion(nu, v_old, dx, dy)

        # buoyancy
        buoyancy = (1. - beta * (h - h0)) * (- grav)

        # intermediate velocity
        u_hat[2:-2, 2:-2] = u_old[2:-2, 2:-2] + dt * (- u_advc + u_diff)
        v_hat[2:-2, 2:-2] = v_old[2:-2, 2:-2] + dt * (- v_advc + v_diff + buoyancy[2:-2, 2:-2])

        # pressure poisson eq
        div_hat = operators_2D.velocity_divergence(u_hat, v_hat, dx, dy)
        b = rho / dt * div_hat
        for it_ppe in range(maxiter_ppe+1):
            p_old = np.copy(p)
            p[1:-1, 1:-1] = 1. / (2. * (dx**2 + dy**2)) \
                            * (
                                - b[1:-1, 1:-1] * dx**2 * dy**2 \
                                + (p_old[2:, 1:-1] + p_old[:-2, 1:-1]) * dy**2 \
                                + (p_old[1:-1, 2:] + p_old[1:-1, :-2]) * dx**2
                            )

            if bc == "natural" or bc == "cold":
                # Neumann boundary condition
                # p[:, -1] = 0.   # North
                p[:, -1] = p[:, -2]   # North
                p[:,  0] = p[:,  1]   # South
                p[-1, :] = p[-2, :]   # East
                p[0,  :] = p[1,  :]   # West
                # p[1,  1] = 0.         # bottom left corner

            elif bc == "periodic":
                # periodic boundary condition
                p[:, -1] = p[:, -2]   # North
                p[:,  0] = p[:,  1]   # South
                p[-1, :] = p[1,  :]   # East = West interior
                p[0,  :] = p[-2, :]   # West = East interior

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

        p_x, p_y = operators_2D.pressure_gradient(p, dx, dy)
        u[2:-2, 2:-2] = u_hat[2:-2, 2:-2] + dt * (- p_x / rho)
        v[2:-2, 2:-2] = v_hat[2:-2, 2:-2] + dt * (- p_y / rho)

        if bc == "natural" or bc == "cold":
            # no-slip boundary condition
            u[:, -2:], v[:, -2:] = 0., 0.   # North
            u[:, :2] , v[:, :2]  = 0., 0.   # South
            u[-2:, :], v[-2:, :] = 0., 0.   # East
            u[:2, :] , v[:2, :]  = 0., 0.   # West

        elif bc == "periodic":
            # periodic boundary condition
            u[:, -2:], v[:, -2:] = 0., 0.   # North
            u[:, :2] , v[:, :2]  = 0., 0.   # South
            u[-1, :] , v[-1, :]  = u[3,  :], v[3,  :]   # East
            u[-2, :] , v[-2, :]  = u[2,  :], v[2,  :]   # East
            u[0,  :] , v[0,  :]  = u[-4, :], v[-4, :]   # West
            u[1,  :] , v[1,  :]  = u[-3, :], v[-3, :]   # West

        # convergence?
        u_flat = u.flatten()
        v_flat = v.flatten()
        u_old_flat = u_old.flatten()
        v_old_flat = v_old.flatten()
        res_u = np.linalg.norm(u_flat - u_old_flat, 2) / np.linalg.norm(u_old_flat, 2)
        res_v = np.linalg.norm(v_flat - v_old_flat, 2) / np.linalg.norm(v_old_flat, 2)
        res_vel = max(res_u, res_v)
        # if res_vel < tol_vel:
        #     print(f"[MAIN] converged")
        #     break
        print(f"\n****************************************************************")
        print(f"[MAIN] t: {t:.3f} / {T:.3f}")
        print(f"[MAIN] it_vel: {it_vel:06d} / {maxiter_vel:06d}")
        print(f"[MAIN] dx: {dx:.3e}, dt: {dt:.3e}")
        print(f"[MAIN] res_vel: {res_vel:.6e} / {tol_vel:.6e}")

        C = np.max(np.abs(u) * dt / dx + np.abs(v) * dt / dy)
        D = np.max(nu * dt / dx**2 + nu * dt / dy**2)
        K = np.max(kap * dt / dx**2 + kap * dt / dy**2)
        print(f"[MAIN] Courant number    : {C:.6f} < 1.0")
        print(f"[MAIN] diffusion number  : {D:.6f} < 0.5")
        print(f"[MAIN] diffusivity number: {K:.6f} < 0.5")

        vel_norm = np.sqrt(u**2 + v**2)
        Re = np.max(vel_norm * Lx / nu)
        print(f"[MAIN] Reynolds number: {Re:.3e}")
        print(f"[MAIN] Prandtl number : {Pr:.3e}")
        print(f"[MAIN] Grashof number : {Gr:.3e}")
        print(f"[MAIN] Rayleigh number: {Ra:.3e}")
        print(f"****************************************************************")

        # print(f"[MAIN] dt: {dt:.3e}")
        # dt1 = 1. * dx**1 / (np.max(vel_norm) * n_dims)
        # dt2 = .5 * dx**2 / (nu * n_dims)
        # dt3 = .5 * dx**2 / (kap * n_dims)
        # dt = min(dt1, dt2, dt3)
        # dt *= .4
        # print(f"[MAIN] dt1: {dt1:.3e}")
        # print(f"[MAIN] dt2: {dt2:.3e}")
        # print(f"[MAIN] dt3: {dt3:.3e}")
        # print(f"[MAIN] dt: {dt:.3e}")

        ########################################################################
        # temperature
        h_old = np.copy(h)
        h_advc = operators_2D.advection(u, v, h, dx, dy, dt, scheme="KK")
        h_diff = operators_2D.diffusion(kap, h, dx, dy)
        h[2:-2, 2:-2] = h_old[2:-2, 2:-2] + dt * (- h_advc + h_diff)

        if bc == "natural":
            # Neumann boundary condition
            h[:, -2:] = h0   # North
            h[:, :2]  = h1   # South
            h[-2:, :] = h[-3, :]   # East
            h[:2, :]  = h[2, :]    # West

            # # Neumann boundary condition
            # h[:, -2:] = h[:, -3:-2]   # North
            # h[:, :2]  = h[:, 2:3]   # South
            # h[-2:, :] = h0   # East
            # h[:2, :]  = h1   # West

        elif bc == "cold":
            # Dirichlet boundary condition
            h[:, -2:] = h0   # North
            h[:, :2]  = h1   # South
            h[-2:, :] = h0   # East
            h[:2, :]  = h0   # West

        elif bc == "periodic":
            # periodic boundary condition
            h[:, -2:] = h0   # North
            h[:, :2]  = h1   # South
            h[-1, :]  = h[3,  :]   # East
            h[-2, :]  = h[2,  :]   # East
            h[0,  :]  = h[-4, :]   # West
            h[1,  :]  = h[-3, :]   # West

        ########################################################################

        if it_vel % int(dump_out_interval / dt) == 0:
            fig, ax = plt.subplots(figsize=(7, 5))

            levels = np.linspace(h0, h1, 32)
            ticks = np.linspace(h0, h1, 5)
            cf = ax.contourf(
                x_vel, y_vel, h,
                levels=levels, cmap="turbo", extend="both"
            )
            if Lx > Ly:
                cb = fig.colorbar(cf, ax=ax, ticks=ticks, orientation="horizontal")
            else:
                cb = fig.colorbar(cf, ax=ax, ticks=ticks, orientation="vertical")

            # vel_norm = np.sqrt(u**2 + v**2)
            # vmin, vmax = np.min(vel_norm), np.max(vel_norm)
            # levels = np.linspace(vmin, vmax, 32)
            # ticks = np.linspace(vmin, vmax, 5)
            # cf = ax.contourf(
            #     x_vel, y_vel, vel_norm,
            #     levels=levels, cmap="turbo", extend="both"
            # )
            # cb = fig.colorbar(cf, ax=ax, ticks=ticks, orientation="horizontal")

            # p_bar = p - np.mean(p)
            # vmin, vmax = np.min(p_bar), np.max(p_bar)
            # levels = np.linspace(vmin, vmax, 32)
            # ticks = np.linspace(vmin, vmax, 5)
            # cf = ax.contourf(
            #     x_prs, y_prs, p_bar,
            #     levels=levels, cmap="seismic", extend="both"
            # )
            # cb = fig.colorbar(cf, ax=ax, ticks=ticks, orientation="horizontal")

            # u_norm = u / vel_norm
            # v_norm = v / vel_norm
            # ax.quiver(
            #     x_vel[1:-1:Nx_vel//20, 1:-1:Ny_vel//20], y_vel[1:-1:Nx_vel//20, 1:-1:Ny_vel//20],
            #     u_norm[1:-1:Nx_vel//20, 1:-1:Ny_vel//20], v_norm[1:-1:Nx_vel//20, 1:-1:Ny_vel//20],
            #     # u[1:-1:Nx_vel//20, 1:-1:Ny_vel//20], v[1:-1:Nx_vel//20, 1:-1:Ny_vel//20],
            #     color="w", pivot="mid"
            # )
            ax.set(
                xticks=np.linspace(0., Lx, 3),
                yticks=np.linspace(0., Ly, 3),
                xlim=(0., Lx),
                ylim=(0., Ly),
                xlabel=rf"$x \text{{ [m]}}$",
                ylabel=rf"$y \text{{ [m]}}$",
                # title=rf"Velocity",
                # title=rf"Temperature",
                # title=rf"$t={t:.3f} / {T:.3f} \text{{ [s]}}$",
                title=rf"$t={t:.3f} \text{{ [s]}}$",
                aspect="equal",
            )

            fig.tight_layout()
            fig.savefig(path_res / f"res.png", bbox_inches="tight")
            fig.savefig(path_res / f"res_t{t:.3f}.png", bbox_inches="tight")
            # fig.savefig(path_res / f"res_it{it_vel:06d}.png")
            plt.close(fig)

        # ########################################################################

        # if it_vel % 200 == 0:
        #     fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        #     # vel_norm = np.sqrt(u**2 + v**2)
        #     # levels = np.linspace(0., Ux, 32)
        #     # ticks = np.linspace(0., Ux, 5)
        #     # cf = ax[0].contourf(
        #     #     x_vel, y_vel, vel_norm,
        #     #     levels=levels, cmap="turbo", extend="both"
        #     # )
        #     # cb = fig.colorbar(cf, ax=ax[0], ticks=ticks, orientation="horizontal")

        #     levels = np.linspace(h0, h1, 32)
        #     ticks = np.linspace(h0, h1, 5)
        #     cf = ax[0].contourf(
        #         x_vel, y_vel, h,
        #         levels=levels, cmap="turbo", extend="both"
        #     )
        #     cb = fig.colorbar(cf, ax=ax[0], ticks=ticks, orientation="horizontal")

        #     u_norm = u / vel_norm
        #     v_norm = v / vel_norm
        #     ax[0].quiver(
        #         x_vel[1:-1:Nx_vel//20, 1:-1:Ny_vel//20], y_vel[1:-1:Nx_vel//20, 1:-1:Ny_vel//20],
        #         u_norm[1:-1:Nx_vel//20, 1:-1:Ny_vel//20], v_norm[1:-1:Nx_vel//20, 1:-1:Ny_vel//20],
        #         # u[1:-1:Nx_vel//20, 1:-1:Ny_vel//20], v[1:-1:Nx_vel//20, 1:-1:Ny_vel//20],
        #         color="w", pivot="mid"
        #     )

        #     ax[0].set(
        #         xticks=np.linspace(0., Lx, 3),
        #         yticks=np.linspace(0., Ly, 3),
        #         xlim=(0., Lx),
        #         ylim=(0., Ly),
        #         xlabel=rf"$x$",
        #         ylabel=rf"$y$",
        #         # title=rf"Velocity",
        #         title=rf"Temperature",
        #         aspect="equal",
        #     )

        #     # levels = np.linspace(h0, h1, 32)
        #     # ticks = np.linspace(h0, h1, 5)
        #     # cf = ax[1].contourf(
        #     #     x_vel, y_vel, h,
        #     #     levels=levels, cmap="turbo", extend="both"
        #     # )
        #     # cb = fig.colorbar(cf, ax=ax[1], ticks=ticks, orientation="horizontal")

        #     p_bar = p
        #     # p_bar = p - np.mean(p)

        #     # dyn_prs = 1./2. * rho * vel_norm**2
        #     # p_bar = p / np.max(dyn_prs)
        #     # p_bar -= np.mean(p_bar)

        #     print(f"p_bar.min(): {p_bar.min():.3e}")
        #     print(f"p_bar.max(): {p_bar.max():.3e}")

        #     levels = np.linspace(-100., 0., 32)
        #     ticks = np.linspace(-100., 0., 5)
        #     # levels = np.linspace(-.2, .2, 32)
        #     # ticks = np.linspace(-.2, .2, 5)
        #     cf = ax[1].contourf(
        #         x_prs, y_prs, p_bar,
        #         levels=levels, cmap="Blues_rf", extend="both"
        #         # levels=levels, cmap="RdBu_rf", extend="both"
        #     )
        #     cb = fig.colorbar(cf, ax=ax[1], ticks=ticks, orientation="horizontal")

        #     ax[1].set(
        #         xticks=np.linspace(0., Lx, 3),
        #         yticks=np.linspace(0., Ly, 3),
        #         xlim=(0., Lx),
        #         ylim=(0., Ly),
        #         xlabel=rf"$x$",
        #         ylabel=rf"$y$",
        #         # title=rf"Temperature",
        #         title=rf"Pressure",
        #         aspect="equal",
        #     )

        #     fig.suptitle(rf"$t={t:.3f} / {T:.3f} \text{{ [s]}}$")
        #     fig.tight_layout()
        #     fig.savefig(path_res / f"res.png")
        #     fig.savefig(path_res / f"res_it{it_vel:06d}.png")
        #     plt.close(fig)

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
    # plt.rcParams['axes.axisbelow'] = True   # background grid
    plt.rcParams["grid.alpha"] = .3
    plt.rcParams["legend.framealpha"] = .8
    plt.rcParams["legend.facecolor"] = "w"
    plt.rcParams["savefig.dpi"] = 300

################################################################################

if __name__ == "__main__":
    plot_setting()
    main()
