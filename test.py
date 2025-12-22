import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

################################################################################

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--Pr", type=float, default=.7, help="Prandtl number")
parser.add_argument("-r", "--Ra", type=float, default=1e5, help="Rayleigh number")
args = parser.parse_args()

################################################################################

def main():
    # path
    path_res = Path("test_results")
    path_res.mkdir(exist_ok=True)

    # パラメータ設定
    # Nx, Ny = 51, 51
    Nx, Ny = 101, 101
    Lx, Ly = 1.0, 1.0
    dx, dy = Lx / (Nx-1), Ly / (Ny-1)

    x = np.linspace(0., Lx, Nx)
    y = np.linspace(0., Ly, Ny)
    # x = np.linspace(dx/2, Lx - dx/2, Nx)
    # y = np.linspace(dy/2, Ly - dy/2, Ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    dt = 0.001
    maxiter = int(2e4)  # タイムステップ数
    beta = 1.0  # 温度膨張率
    Pr = args.Pr
    Ra = args.Ra
    Re = np.sqrt(Ra / Pr)  # 有効なRe

    # フィールド変数の初期化
    u = np.zeros((Nx, Ny)) + .0 * np.random.normal(loc=0.0, scale=1.0, size=(Nx, Ny))
    v = np.zeros((Nx, Ny)) + .0 * np.random.normal(loc=0.0, scale=1.0, size=(Nx, Ny))
    p = np.zeros((Nx, Ny)) + .0 * np.random.normal(loc=0.0, scale=1.0, size=(Nx, Ny))
    T = np.zeros((Nx, Ny)) + .0 * np.random.normal(loc=0.0, scale=1.0, size=(Nx, Ny))

    # 初期温度（床 = 1, 天井 = 0）
    T[:, 0] = 1.0
    T[:, -1] = 0.0

    # T[ 0, :] = 1.  # 左の壁
    # T[-1, :] = 0.  # 右の壁

    # ヘルパー関数（前ステップのコード再利用）
    def compute_intermediate_velocity(u, v, p, T, dx, dy, dt, Re, beta):
        u_star = u.copy()
        v_star = v.copy()

        for i in range(1, Nx-1):
            for j in range(1, Ny-1):
                du2dx = (u[i+1,j]**2 - u[i-1,j]**2) / (2*dx)
                duvdy = ((u[i,j+1]+u[i,j])*(v[i+1,j]+v[i,j]) - (u[i,j]+u[i,j-1])*(v[i+1,j-1]+v[i,j-1])) / (4*dy)
                d2udx2 = (u[i+1,j] - 2*u[i,j] + u[i-1,j]) / dx**2
                d2udy2 = (u[i,j+1] - 2*u[i,j] + u[i,j-1]) / dy**2

                u_star[i,j] = u[i,j] + dt * (
                    -du2dx - duvdy +
                    (1/Re) * (d2udx2 + d2udy2)
                )

                # 温度による浮力項は y 方向の運動方程式に影響
                dv2dy = (v[i,j+1]**2 - v[i,j-1]**2) / (2*dy)
                duvdx = ((u[i+1,j]+u[i,j])*(v[i,j+1]+v[i,j]) - (u[i-1,j]+u[i,j])*(v[i,j]+v[i,j-1])) / (4*dx)
                d2vdx2 = (v[i+1,j] - 2*v[i,j] + v[i-1,j]) / dx**2
                d2vdy2 = (v[i,j+1] - 2*v[i,j] + v[i,j-1]) / dy**2

                v_star[i,j] = v[i,j] + dt * (
                    -dv2dy - duvdx +
                    (1/Re) * (d2vdx2 + d2vdy2) +
                    beta * T[i,j]
                )
        return u_star, v_star

    def pressure_poisson_jacobi(p, u_star, v_star, dx, dy, dt, nit=int(1e3)):
        pn = p.copy()
        for _ in range(nit):
            p[1:-1,1:-1] = (
                (dy**2 * (pn[2:,1:-1] + pn[:-2,1:-1]) +
                dx**2 * (pn[1:-1,2:] + pn[1:-1,:-2]) -
                dx**2 * dy**2 / dt * (
                    (u_star[2:,1:-1] - u_star[:-2,1:-1]) / (2*dx) +
                    (v_star[1:-1,2:] - v_star[1:-1,:-2]) / (2*dy)
                ))
                / (2 * (dx**2 + dy**2))
            )
            pn = p.copy()
            p[:, -1] = p[:, -2]
            p[:,  0] = p[:,  1]
            p[ 0, :] = p[ 1, :]
            p[-1, :] = p[-2, :]
            # p[:, -1] = 0.
        return p

    def update_velocity(u_star, v_star, p, dx, dy, dt):
        u_new = u_star.copy()
        v_new = v_star.copy()

        u_new[1:-1,1:-1] -= dt * (p[2:,1:-1] - p[:-2,1:-1]) / (2*dx)
        v_new[1:-1,1:-1] -= dt * (p[1:-1,2:] - p[1:-1,:-2]) / (2*dy)

        return u_new, v_new

    def update_temperature(T, u, v, dx, dy, dt, Re, Pr):
        T_new = T.copy()
        alpha = 1.0 / (Re * Pr)
        for i in range(1, Nx-1):
            for j in range(1, Ny-1):
                adv_x = u[i, j] * (T[i, j] - T[i-1, j]) / dx if u[i, j] > 0 else u[i, j] * (T[i+1, j] - T[i, j]) / dx
                adv_y = v[i, j] * (T[i, j] - T[i, j-1]) / dy if v[i, j] > 0 else v[i, j] * (T[i, j+1] - T[i, j]) / dy
                diff_x = (T[i+1, j] - 2*T[i, j] + T[i-1, j]) / dx**2
                diff_y = (T[i, j+1] - 2*T[i, j] + T[i, j-1]) / dy**2
                T_new[i, j] = T[i, j] + dt * (-adv_x - adv_y + alpha * (diff_x + diff_y))

        # 境界条件
        T_new[:,  0] = 1.  # 床を温める
        T_new[:, -1] = 0.  # 天井を冷やす
        T_new[ 0, :] = T_new[1, :]
        T_new[-1, :] = T_new[-2, :]

        # # 境界条件
        # T_new[:,  0] = T_new[:,  1]  # 床
        # T_new[:, -1] = T_new[:, -2]  # 天井
        # T_new[ 0, :] = 1.  # 左の壁
        # T_new[-1, :] = 0.  # 右の壁
        return T_new

    # 🔁 時間ループ
    start_time = datetime.now()
    for it in range(maxiter):
        print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}", end="")
        print(f"\r{it}/{maxiter}", end="")
        u_star, v_star = compute_intermediate_velocity(u, v, p, T, dx, dy, dt, Re, beta)
        p = pressure_poisson_jacobi(p, u_star, v_star, dx, dy, dt)
        u, v = update_velocity(u_star, v_star, p, dx, dy, dt)
        T = update_temperature(T, u, v, dx, dy, dt, Re, Pr)

        if it % 100 == 0:
            fig = plt.figure(figsize=(10, 4))

            ax = fig.add_subplot(121)
            levels = np.linspace(0., 1., 32)
            ticks = np.linspace(0., 1., 5)
            cf = ax.contourf(X, Y, T, levels=levels, cmap="magma")
            fig.colorbar(cf, ticks=ticks, label="$T$")
            vel_norm = np.sqrt(u**2 + v**2)
            step = 5
            qv = ax.quiver(
                X[::step, ::step], Y[::step, ::step],
                u[::step, ::step]/vel_norm[::step, ::step], v[::step, ::step]/vel_norm[::step, ::step],
                T[::step, ::step],
                cmap="magma_r"
            )
            ax.set(
                xticks=np.linspace(0., Lx, 5),
                yticks=np.linspace(0., Lx, 5),
                xlabel=r"$x$",
                ylabel=r"$y$",
                title=rf"Temperature",
                aspect="equal",
            )

            ax = fig.add_subplot(122)
            levels = np.linspace(-.5, .5, 32)
            ticks = np.linspace(-.5, .5, 5)
            cf = ax.contourf(X, Y, p - np.mean(p), levels=levels, cmap="twilight_shifted")
            fig.colorbar(cf, ticks=ticks, label="$p$")
            ax.set(
                xticks=np.linspace(0., Lx, 5),
                yticks=np.linspace(0., Lx, 5),
                xlabel=r"$x$",
                ylabel=r"$y$",
                title=rf"Pressure",
                aspect="equal",
            )

            fig.suptitle(rf"$(\mathrm{{Pr}}, \mathrm{{Ra}}) = ({Pr:.1f}, {Ra:.1e}), \ t = {dt * it:.1f}/{dt * maxiter:.1f}$")
            fig.tight_layout()
            fig.savefig(path_res / f"aaa.png")
            # fig.savefig(path_res / f"it_{it:06d}.png")
            plt.close(fig)


################################################################################

def plot_setting():
    plt.style.use("default")
    plt.style.use("seaborn-v0_8-deep")
    plt.style.use("seaborn-v0_8-talk")
    plt.rcParams["font.family"] = "STIXGeneral"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["figure.figsize"] = (7, 5)
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = .3
    plt.rcParams["legend.framealpha"] = 1.
    plt.rcParams["legend.facecolor"] = "w"
    plt.rcParams["savefig.dpi"] = 300

################################################################################

if __name__ == "__main__":
    plot_setting()
    main()
