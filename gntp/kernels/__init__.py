# -*- coding: utf-8 -*-

from gntp.kernels.base import BaseKernel

from gntp.kernels.linear import LinearKernel
from gntp.kernels.rbf import RBFKernel
from gntp.kernels.cosine import CosineKernel


def get_kernel_by_name(name, slope=1.0):
    name_to_kernel_class = {
        'linear': LinearKernel,
        'rbf': RBFKernel,
        'cosine': CosineKernel
    }

    if name not in name_to_kernel_class:
        raise ModuleNotFoundError

    kernel_class = name_to_kernel_class[name]
    kwargs = {'slope': slope} if name in {'rbf'} else {}
    kernel = kernel_class(**kwargs)
    return kernel


__all__ = [
    'BaseKernel',
    'LinearKernel',
    'RBFKernel',
    'CosineKernel',
    'get_kernel_by_name'
]
