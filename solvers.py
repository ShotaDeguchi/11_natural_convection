"""
********************************************************************************
iterative solvers for linear systems
********************************************************************************
"""

import numpy as np


def Jacobi(A, b, x, tol=1e-9, ord=np.inf, max_iter=int(1e3), verbose=True):
    print("\n=== Jacobi method ===")
    # decompose: A = D - (L + U)
    D = np.diag(np.diag(A))
    LU = D - A
    M = np.dot(np.linalg.inv(D), LU)
    c = np.dot(np.linalg.inv(D), b)

    # cond = np.linalg.cond(A, p=2)   # condition number
    # rho = np.max(np.abs(np.linalg.eigvals(M)))   # spectral radius
    # print(f">>> cond(A): {cond:.6e}")
    # print(f">>> rho(M): {rho:.6e}")

    it_log = []
    res_log = []
    for it in range(0, max_iter+1):
        x = np.dot(M, x) + c
        r = b - np.dot(A, x)
        res = np.linalg.norm(r, ord=ord) / np.linalg.norm(b, ord=ord)
        if it % 100 == 0:
            if verbose:
                print(f">>> it: {it:d}, res: {res:.6e}")
            it_log.append(it)
            res_log.append(res)
        if res < tol:
            print(f">>> converged")
            break
    return x, it_log, res_log


def DampedJacobi(A, b, x, omega=.8, tol=1e-9, ord=np.inf, max_iter=int(1e3), verbose=True):
    print("\n=== Damped Jacobi method ===")
    # decompose: A = D - (L + U)
    D = np.diag(np.diag(A))
    LU = D - A
    M = omega * np.dot(np.linalg.inv(D), LU) + (1. - omega) * np.eye(x.shape[0])
    c = omega * np.dot(np.linalg.inv(D), b)

    # cond = np.linalg.cond(A, p=2)   # condition number
    # rho = np.max(np.abs(np.linalg.eigvals(M)))   # spectral radius
    # print(f">>> cond(A): {cond:.6e}")
    # print(f">>> rho(M): {rho:.6e}")

    it_log = []
    res_log = []
    for it in range(0, max_iter+1):
        x = np.dot(M, x) + c
        r = b - np.dot(A, x)
        res = np.linalg.norm(r, ord=ord)
        if it % 100 == 0:
            if verbose:
                print(f">>> it: {it:d}, res: {res:.6e}")
            it_log.append(it)
            res_log.append(res)
        if res < tol:
            print(f">>> converged")
            break
    return x, it_log, res_log


def GaussSeidel(A, b, x, tol=1e-9, ord=np.inf, max_iter=int(1e3), verbose=True):
    print("\n=== Gauss-Seidel method ===")
    # decompose: A = D - (L + U)
    D = np.diag(np.diag(A))
    L = - np.tril(A, k=-1)
    U = - np.triu(A, k=1)
    M = np.dot(np.linalg.inv(D - L), U)
    c = np.dot(np.linalg.inv(D - L), b)

    # cond = np.linalg.cond(A, p=2)   # condition number
    # rho = np.max(np.abs(np.linalg.eigvals(M)))   # spectral radius
    # print(f">>> cond(A): {cond:.6e}")
    # print(f">>> rho(M): {rho:.6e}")

    it_log = []
    res_log = []
    for it in range(0, max_iter+1):
        x = np.dot(M, x) + c
        r = b - np.dot(A, x)
        res = np.linalg.norm(r, ord=ord)
        if it % 100 == 0:
            if verbose:
                print(f">>> it: {it:d}, res: {res:.6e}")
            it_log.append(it)
            res_log.append(res)
        if res < tol:
            print(f">>> converged")
            break
    return x, it_log, res_log


def SOR(A, b, x, omega=1.8, tol=1e-9, ord=np.inf, max_iter=int(1e3), verbose=True, optimal=False):
    print("\n=== Successive Over-Relaxation method ===")
    # decompose: A = D - (L + U)
    D = np.diag(np.diag(A))
    L = - np.tril(A, k=-1)
    U = - np.triu(A, k=1)
    M = np.dot(np.linalg.inv(D - omega * L), (1. - omega) * D + omega * U)
    c = omega * np.dot(np.linalg.inv(D - omega * L), b)

    # cond = np.linalg.cond(A, p=2)   # condition number
    # rho = np.max(np.abs(np.linalg.eigvals(M)))   # spectral radius
    # print(f">>> cond(A): {cond:.6e}")
    # print(f">>> rho(M): {rho:.6e}")

    if optimal:
        # optimal omega based on spectral radius of Jacobi method
        LU = D - A
        M_J = np.dot(np.linalg.inv(D), LU)
        rho_J = np.max(np.abs(np.linalg.eigvals(M_J)))
        omega_opt = 2. / (1. + np.sqrt(1. - rho_J**2))
        print(f">>> chosen omega : {omega:.6f}")
        print(f">>> optimal omega: {omega_opt:.6f}")
        omega = omega_opt

    it_log = []
    res_log = []
    for it in range(0, max_iter+1):
        x = np.dot(M, x) + c
        r = b - np.dot(A, x)
        res = np.linalg.norm(r, ord=ord)
        if it % 100 == 0:
            if verbose:
                print(f">>> it: {it:d}, res: {res:.6e}")
            it_log.append(it)
            res_log.append(res)
        if res < tol:
            print(f">>> converged")
            break
    return x, it_log, res_log


def SteepDesc(A, b, x, tol=1e-9, ord=np.inf, max_iter=int(1e3), verbose=True):
    print("\n=== Steepest Descent method ===")
    r = b - np.dot(A, x)
    it_log = []
    res_log = []
    alpha_log = []
    # print(f">>> dim(x): {x.shape[0]:d}")
    # print(f">>> cond(A): {np.linalg.cond(A):.6e}")
    for it in range(0, max_iter+1):
        Ar = np.dot(A, r)
        alpha = np.dot(r, r) / np.dot(r, Ar)
        x += alpha * r
        r -= alpha * Ar
        res = np.linalg.norm(r, ord=ord)
        if it % 100 == 0:
            if verbose:
                print(f">>> it: {it:d}, res: {res:.6e}")
            it_log.append(it)
            res_log.append(res)
            alpha_log.append(alpha)
        if res < tol:
            print(f">>> converged")
            break
    return x, it_log, res_log


def ConjDir(A, b, x, tol=1e-9, ord=np.inf, max_iter=int(1e3), verbose=True):
    print("\n=== Conjugate Direction method ===")
    raise NotImplementedError


def ConjGrad(A, b, x, tol=1e-9, ord=np.inf, max_iter=int(1e3), verbose=True):
    print("\n=== Conjugate Gradient method ===")
    r0 = b - np.dot(A, x)
    p = np.copy(r0)
    it_log = []
    res_log = []
    alpha_log = []
    # print(f">>> dim(x): {x.shape[0]:d}")
    # print(f">>> cond(A): {np.linalg.cond(A):.6e}")
    for it in range(0, max_iter+1):
        Ap = np.dot(A, p)
        alpha = np.dot(r0, r0) / np.dot(p, Ap)
        x += alpha * p
        r1 = r0 - alpha * Ap
        res = np.linalg.norm(r1, ord=ord)
        if it % 100 == 0:
            if verbose:
                if verbose:
                    print(f">>> it: {it:d}, res: {res:.6e}")
            it_log.append(it)
            res_log.append(res)
            alpha_log.append(alpha)
        if res < tol:
            print(f">>> converged")
            break
        beta = np.dot(r1, r1) / np.dot(r0, r0)
        p = r1 + beta * p
        r0 = np.copy(r1)
    return x, it_log, res_log

