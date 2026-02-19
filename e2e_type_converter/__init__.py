"""EveryThing to EveryThing 类型转换兼容层

This module provides a comprehensive type conversion compatibility layer for Python,
supporting bidirectional conversion between all standard data types and third-party library types.

主要功能：
- 支持所有Python标准数据类型之间的转换
- 支持第三方库类型（numpy、cupy、scipy、pandas、torch、xarray、jax、tensorflow）
- 提供无感转换机制，保持与Python内置函数的兼容性
- 支持第三方库类型之间的互转
- 类型安全，提供明确的错误提示

示例用法：
    >>> from e2e_type_converter import e2e_list, e2e_str, e2e_int
    >>> e2e_list(123)  # 输出: [123]
    >>> e2e_str(None)  # 输出: ""
    >>> e2e_int("123")  # 输出: 123

    >>> from e2e_type_converter import TypeConverter
    >>> import numpy as np
    >>> arr = np.array([1, 2, 3])
    >>> TypeConverter.numpy_to_xarray(arr)  # 转换为xarray DataArray
"""

from .core import (
    TypeConverter,
    e2e_list,
    e2e_str,
    e2e_int,
    e2e_float,
    e2e_dict,
    e2e_set,
    e2e_tuple,
)

__all__ = [
    'TypeConverter',
    'e2e_list',
    'e2e_str',
    'e2e_int',
    'e2e_float',
    'e2e_dict',
    'e2e_set',
    'e2e_tuple',
]

__version__ = "0.1.1"
__author__ = "John-is-playing"
__email__ = "b297209694@outlook.com"
__license__ = "MIT"
