"""
********************************************************************************
natural convection in a rectangular cavity
********************************************************************************
"""

import glob
import time
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import skimage

import operators_3D

################################################################################

def main():
    # seed
    np.random.seed(42)

    # domain
    Lx, Ly, Lz = 2., 2., 1.
    h_res = 2e-2
    dx, dy, dz = h_res, h_res, h_res

    # # arakawa-b grid
    # x_vel, y_vel, z_vel = np.arange(0.-dx, Lx+2.*dx, dx), np.arange(0.-dy, Ly+2.*dy, dy), np.arange(0.-dz, Lz+2.*dz, dz)
    # Nx_vel, Ny_vel, Nz_vel = len(x_vel), len(y_vel), len(z_vel)
    # x_vel, y_vel, z_vel = np.meshgrid(x_vel, y_vel, z_vel, indexing="ij")

    # x_prs, y_prs, z_prs = np.arange(0.-1./2.*dx, Lx+3./2.*dx, dx), np.arange(0.-1./2.*dy, Ly+3./2.*dy, dy), np.arange(0.-1./2.*dz, Lz+3./2.*dz, dz)
    # Nx_prs, Ny_prs, Nz_prs = len(x_prs), len(y_prs), len(z_prs)
    # x_prs, y_prs, z_prs = np.meshgrid(x_prs, y_prs, z_prs, indexing="ij")

    # boundary temperature
    h0, h1 = 20., 100.
    h0, h1 = h0 + 273.15, h1 + 273.15
    theta = h1 - h0
    # theta = (h1 - h0) / Ly

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
    Gr = (grav * beta * Lz**3 * theta) / nu**2
    Ra = Pr * Gr

    # boundary condition
    bc = "periodic"  # "natural" / "periodic"

    # path
    path_res = Path(f"res_3D_Pr{Pr:.3e}_Ra{Ra:.3e}_{bc}")
    # path_fig = path_res / "fig"
    # path_fig.mkdir(parents=True, exist_ok=True)
    path_npz = path_res / "npz"
    # path_npz.mkdir(parents=True, exist_ok=True)
    path_vis = path_res / "vis"
    path_vis.mkdir(parents=True, exist_ok=True)

    # load
    data = np.load(path_npz / "res_it000000.npz")
    # print(f"data: {data.files}")

    # get all npz files
    files = sorted(glob.glob(str(path_npz / "res_it*.npz")))
    # print(f"files: {files}")

    # loop through all files
    for idx, file in enumerate(files):
        if idx % 1 == 0:
            print(f"Processing file {idx+1}/{len(files)}: {file}")
            data = np.load(file)
            u = data["u"]
            v = data["v"]
            w = data["w"]
            p = data["p"]
            h = data["h"]
            x_vel = data["x_vel"]
            y_vel = data["y_vel"]
            z_vel = data["z_vel"]
            x_prs = data["x_prs"]
            y_prs = data["y_prs"]
            z_prs = data["z_prs"]
            Lx = data["Lx"]
            Ly = data["Ly"]
            Lz = data["Lz"]
            dx = data["dx"]
            dy = data["dy"]
            dz = data["dz"]
            t = data["t"]

            # visualize
            mask = (x_vel < Lx/2.) | (y_vel > Ly/2.)
            h_masked = np.where(mask, h, np.nan)

            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

            iso_vals = np.linspace(h0, h1, 5)
            iso_vals = np.linspace(h0+10., h1-10., 5)
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
            fig.savefig(path_vis / f"vis.png")
            fig.savefig(path_vis / f"vis_t{t:.3f}.png")
            plt.close(fig)

            # visualize the histogram of temperature
            fig, ax = plt.subplots()
            h_flat = h.flatten()
            ax.hist(h_flat, bins=64, color="C0", alpha=.7, density=True, range=(h0, h1))
            ax.set(
                xticks=np.linspace(293.15, 373.15, 5),
                xlim=(293.15-10., 373.15+10.),
                ylim=(0., .06),
                xlabel=rf"$T \text{{ [K]}}$",
                ylabel=rf"$\text{{Density}}$",
                title=rf"$t={t:.3f} \text{{ [s]}}$",
            )
            fig.tight_layout()
            fig.savefig(path_vis / f"vis_hist_t{t:.3f}.png")
            plt.close(fig)

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
