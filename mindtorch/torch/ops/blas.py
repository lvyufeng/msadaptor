"""blas op"""
import mindspore

from mindspore import ops
from ..configs import use_pyboost, ON_ORANGE_PI

# addbmm
has_addbmm = hasattr(mindspore.mint, 'addbmm')
def addbmm(input, batch1, batch2, *, beta=1, alpha=1):
    if use_pyboost() and has_addbmm:
        return mindspore.mint.addbmm(input, batch1, batch2, beta=beta, alpha=alpha)
    return ops.addbmm(input, batch1, batch2, beta=beta, alpha=alpha)

# addmm
has_addmm = hasattr(mindspore.mint, 'addmm')
def addmm(input, mat1, mat2, *, beta=1, alpha=1):
    if use_pyboost() and has_addmm:
        return mindspore.mint.addmm(input, mat1, mat2, beta=beta, alpha=alpha)
    return ops.addmm(input, mat1, mat2, beta=beta, alpha=alpha)

# addmv


# addr


# baddbmm
has_baddbmm = hasattr(mindspore.mint, 'baddbmm')
def baddbmm(input, batch1, batch2, *, beta=1, alpha=1):
    if use_pyboost() and has_baddbmm:
        return mindspore.mint.baddbmm(input, batch1, batch2, beta=beta, alpha=alpha)
    return ops.baddbmm(input, batch1, batch2, beta=beta, alpha=alpha)

# bmm
has_bmm = hasattr(mindspore.mint, 'bmm')
def bmm(input, other):
    if ON_ORANGE_PI:
        input = input.to(mindspore.float16)
        other = input.to(mindspore.float16)
    if use_pyboost() and has_bmm:
        return mindspore.mint.bmm(input, other)
    return ops.bmm(input, other)

# chain_matmul


# cholesky

# cholesky_inverse

# cholesky_solve

# dot
has_dot = hasattr(mindspore.mint, 'dot')
def dot(input, other):
    if use_pyboost() and has_dot:
        return mindspore.mint.dot(input, other)
    return (input * other).sum()

# geqrf

# ger

# inner

# inverse

# det

# logdet

# slogdet

# lu

# lu_solve


# lu_unpack

# matmul
has_matmul = hasattr(mindspore.mint, 'matmul')
def matmul(input, other):
    if ON_ORANGE_PI:
        input = input.to(mindspore.float16)
        other = other.to(mindspore.float16)
    if use_pyboost() and has_matmul:
        return mindspore.mint.matmul(input, other)
    return ops.matmul(input, other)

# matrix_power

# matrix_exp

# mm
has_mm = hasattr(mindspore.mint, 'mm')
def mm(input, other):
    return matmul(input, other)

# mv


# orgqr

# ormqr

# outer
has_outer = hasattr(mindspore.mint, 'outer')
def outer(input, vec2):
    if use_pyboost() and has_outer:
        return mindspore.mint.outer(input, vec2)
    return ops.outer(input, vec2)

# pinverse


# qr

# svd

# svd_lowrank

# pca_lowrank


# lobpcg


# trapz


# trapezoid


# cumulative_trapezoid


# triangular_solve


# vdot

__all__ = [
    'addbmm',
    'addmm',
    # addmv
    # addr
    'baddbmm',
    'bmm',
    # chain_matmul
    # cholesky
    # cholesky_inverse
    # cholesky_solve
    'dot',
    # geqrf
    # ger
    # inner
    # inverse
    # det
    # logdet
    # slogdet
    # lu
    # lu_solve
    # lu_unpack
    'matmul',
    # matrix_power
    # matrix_exp
    'mm',
    # mv
    # orgqr
    # ormqr
    'outer',
    # pinverse
    # qr
    # svd
    # svd_lowrank
    # pca_lowrank
    # lobpcg
    # trapz
    # trapezoid
    # cumulative_trapezoid
    # triangular_solve
    # vdot
]
