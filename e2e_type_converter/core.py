"""核心类型转换实现

This module contains the core implementation of the type conversion compatibility layer,
including support for standard Python types and third-party library types.
"""

import builtins
import sys
from functools import lru_cache

# 保存原始内置转换函数
original_list = builtins.list
original_str = builtins.str
original_int = builtins.int
original_float = builtins.float
original_dict = builtins.dict
original_set = builtins.set
original_tuple = builtins.tuple

# 缓存装饰器，用于缓存转换结果
def conversion_cache(func):
    """缓存转换结果的装饰器
    
    Args:
        func: 转换函数
        
    Returns:
        装饰后的函数
    """
    # 创建一个简单的缓存字典
    cache = {}
    
    def cached_func(*args, **kwargs):
        try:
            # 尝试将参数转换为可哈希的键
            key = args + tuple(sorted(kwargs.items()))
            if key in cache:
                return cache[key]
            result = func(*args, **kwargs)
            cache[key] = result
            # 限制缓存大小
            if len(cache) > 1000:
                # 简单的缓存清理策略：移除一半的条目
                keys = list(cache.keys())[:len(cache)//2]
                for k in keys:
                    del cache[k]
            return result
        except TypeError:
            # 如果参数不可哈希，直接调用函数
            return func(*args, **kwargs)
    
    return cached_func

class TypeConverter:
    """类型转换兼容层
    
    A comprehensive type conversion compatibility layer that supports bidirectional
    conversion between all standard Python data types and third-party library types.
    """
    
    # 第三方库类型检测缓存
    _numpy_available = None
    _cupy_available = None
    _scipy_available = None
    _pandas_available = None
    _torch_available = None
    _xarray_available = None
    _jax_available = None
    _tensorflow_available = None
    
    # 第三方库模块缓存
    _numpy = None
    _cupy = None
    _scipy = None
    _pandas = None
    _torch = None
    _xarray = None
    _jax = None
    _tensorflow = None
    
    # 第三方库互转方法
    @staticmethod
    def numpy_to_xarray(obj):
        """将numpy数组转换为xarray DataArray
        
        Convert numpy array to xarray DataArray with automatically generated dimensions and coordinates.
        
        Args:
            obj: numpy.ndarray - 要转换的numpy数组
            
        Returns:
            xarray.DataArray - 转换后的xarray DataArray
            
        Raises:
            TypeError: 如果输入不是numpy数组
        """
        if not TypeConverter._is_numpy_array(obj):
            raise TypeError(f"Expected numpy.ndarray, got {type(obj).__name__}")
        
        numpy = TypeConverter._get_numpy()
        xarray = TypeConverter._get_xarray()
        
        # 创建维度名称
        dims = [f'dim{i}' for i in range(obj.ndim)]
        
        # 创建坐标
        coords = {}
        for i, size in enumerate(obj.shape):
            coords[dims[i]] = numpy.arange(size)
        
        return xarray.DataArray(obj, dims=dims, coords=coords)
    
    @staticmethod
    def xarray_to_numpy(obj):
        """将xarray DataArray转换为numpy数组
        
        Convert xarray DataArray or Dataset to numpy array.
        
        Args:
            obj: xarray.DataArray or xarray.Dataset - 要转换的xarray对象
            
        Returns:
            numpy.ndarray - 转换后的numpy数组
            
        Raises:
            TypeError: 如果输入不是xarray对象
            ValueError: 如果输入是空的xarray Dataset
        """
        if not (TypeConverter._is_xarray_dataarray(obj) or TypeConverter._is_xarray_dataset(obj)):
            raise TypeError(f"Expected xarray.DataArray or xarray.Dataset, got {type(obj).__name__}")
        
        if TypeConverter._is_xarray_dataarray(obj):
            return obj.values
        elif TypeConverter._is_xarray_dataset(obj):
            # 对于Dataset，返回第一个变量的values
            if obj.data_vars:
                return next(iter(obj.data_vars.values())).values
            raise ValueError("Empty xarray Dataset")
    
    @staticmethod
    def torch_to_numpy(obj):
        """将torch Tensor转换为numpy数组
        
        Convert torch Tensor to numpy array, detaching from computation graph if necessary.
        
        Args:
            obj: torch.Tensor - 要转换的torch张量
            
        Returns:
            numpy.ndarray - 转换后的numpy数组
            
        Raises:
            TypeError: 如果输入不是torch张量
        """
        if not TypeConverter._is_torch_tensor(obj):
            raise TypeError(f"Expected torch.Tensor, got {type(obj).__name__}")
        
        return obj.detach().cpu().numpy()
    
    @staticmethod
    def numpy_to_torch(obj):
        """将numpy数组转换为torch Tensor
        
        Convert numpy array to torch Tensor.
        
        Args:
            obj: numpy.ndarray - 要转换的numpy数组
            
        Returns:
            torch.Tensor - 转换后的torch张量
            
        Raises:
            TypeError: 如果输入不是numpy数组
        """
        if not TypeConverter._is_numpy_array(obj):
            raise TypeError(f"Expected numpy.ndarray, got {type(obj).__name__}")
        
        torch = TypeConverter._get_torch()
        return torch.tensor(obj)
    
    @staticmethod
    def jax_to_numpy(obj):
        """将jax数组转换为numpy数组
        
        Convert jax array to numpy array.
        
        Args:
            obj: jax.numpy.ndarray - 要转换的jax数组
            
        Returns:
            numpy.ndarray - 转换后的numpy数组
            
        Raises:
            TypeError: 如果输入不是jax数组
        """
        if not TypeConverter._is_jax_array(obj):
            raise TypeError(f"Expected jax.numpy.ndarray, got {type(obj).__name__}")
        
        return obj.__array__()
    
    @staticmethod
    def numpy_to_jax(obj):
        """将numpy数组转换为jax数组
        
        Convert numpy array to jax array.
        
        Args:
            obj: numpy.ndarray - 要转换的numpy数组
            
        Returns:
            jax.numpy.ndarray - 转换后的jax数组
            
        Raises:
            TypeError: 如果输入不是numpy数组
        """
        if not TypeConverter._is_numpy_array(obj):
            raise TypeError(f"Expected numpy.ndarray, got {type(obj).__name__}")
        
        jax = TypeConverter._get_jax()
        return jax.numpy.array(obj)
    
    @staticmethod
    def tensorflow_to_numpy(obj):
        """将tensorflow Tensor转换为numpy数组
        
        Convert tensorflow Tensor to numpy array.
        
        Args:
            obj: tensorflow.Tensor - 要转换的tensorflow张量
            
        Returns:
            numpy.ndarray - 转换后的numpy数组
            
        Raises:
            TypeError: 如果输入不是tensorflow张量
        """
        if not TypeConverter._is_tensorflow_tensor(obj):
            raise TypeError(f"Expected tensorflow.Tensor, got {type(obj).__name__}")
        
        return obj.numpy()
    
    @staticmethod
    def numpy_to_tensorflow(obj):
        """将numpy数组转换为tensorflow Tensor
        
        Convert numpy array to tensorflow Tensor.
        
        Args:
            obj: numpy.ndarray - 要转换的numpy数组
            
        Returns:
            tensorflow.Tensor - 转换后的tensorflow张量
            
        Raises:
            TypeError: 如果输入不是numpy数组
        """
        if not TypeConverter._is_numpy_array(obj):
            raise TypeError(f"Expected numpy.ndarray, got {type(obj).__name__}")
        
        tensorflow = TypeConverter._get_tensorflow()
        return tensorflow.constant(obj)
    
    @staticmethod
    def pandas_to_numpy(obj):
        """将pandas对象转换为numpy数组
        
        Convert pandas DataFrame or Series to numpy array.
        
        Args:
            obj: pandas.DataFrame or pandas.Series - 要转换的pandas对象
            
        Returns:
            numpy.ndarray - 转换后的numpy数组
            
        Raises:
            TypeError: 如果输入不是pandas对象
        """
        if not (TypeConverter._is_pandas_dataframe(obj) or TypeConverter._is_pandas_series(obj)):
            raise TypeError(f"Expected pandas.DataFrame or pandas.Series, got {type(obj).__name__}")
        
        return obj.values
    
    @staticmethod
    def numpy_to_pandas(obj):
        """将numpy数组转换为pandas对象
        
        Convert numpy array to pandas Series or DataFrame.
        
        Args:
            obj: numpy.ndarray - 要转换的numpy数组
            
        Returns:
            pandas.Series or pandas.DataFrame - 转换后的pandas对象
            
        Raises:
            TypeError: 如果输入不是numpy数组
            ValueError: 如果输入数组维度大于2
        """
        if not TypeConverter._is_numpy_array(obj):
            raise TypeError(f"Expected numpy.ndarray, got {type(obj).__name__}")
        
        pandas = TypeConverter._get_pandas()
        
        if obj.ndim == 1:
            return pandas.Series(obj)
        elif obj.ndim == 2:
            # 创建列名
            columns = [f'col{i}' for i in range(obj.shape[1])]
            return pandas.DataFrame(obj, columns=columns)
        else:
            raise ValueError(f"Cannot convert {obj.ndim}-dimensional numpy array to pandas object")
    
    @staticmethod
    def convert(obj, target_type):
        """通用类型转换方法，支持第三方库类型之间的互转
        
        Generic type conversion method that supports conversion between third-party library types.
        
        Args:
            obj: 要转换的对象
            target_type: 目标类型，可以是以下字符串之一：
                'numpy', 'cupy', 'scipy', 'pandas', 'torch', 'xarray', 'jax', 'tensorflow'
        
        Returns:
            转换后的对象
            
        Raises:
            ValueError: 如果目标类型无效
            TypeError: 如果无法转换输入对象
        """
        # 检查目标类型
        valid_targets = ['numpy', 'cupy', 'scipy', 'pandas', 'torch', 'xarray', 'jax', 'tensorflow']
        if target_type not in valid_targets:
            raise ValueError(f"Invalid target_type: {target_type}. Must be one of {valid_targets}")
        
        # 直接转换路径，避免不必要的中间转换
        # torch -> cupy
        if TypeConverter._is_torch_tensor(obj) and target_type == 'cupy':
            cupy = TypeConverter._get_cupy()
            return cupy.asarray(obj.detach().cpu())
        # cupy -> torch
        elif TypeConverter._is_cupy_array(obj) and target_type == 'torch':
            torch = TypeConverter._get_torch()
            return torch.as_tensor(obj.get())
        # torch -> tensorflow
        elif TypeConverter._is_torch_tensor(obj) and target_type == 'tensorflow':
            tensorflow = TypeConverter._get_tensorflow()
            return tensorflow.convert_to_tensor(obj.detach().cpu().numpy())
        # tensorflow -> torch
        elif TypeConverter._is_tensorflow_tensor(obj) and target_type == 'torch':
            torch = TypeConverter._get_torch()
            return torch.as_tensor(obj.numpy())
        # jax -> torch
        elif TypeConverter._is_jax_array(obj) and target_type == 'torch':
            torch = TypeConverter._get_torch()
            return torch.as_tensor(obj.__array__())
        # torch -> jax
        elif TypeConverter._is_torch_tensor(obj) and target_type == 'jax':
            jax = TypeConverter._get_jax()
            return jax.numpy.array(obj.detach().cpu().numpy())
        # jax -> tensorflow
        elif TypeConverter._is_jax_array(obj) and target_type == 'tensorflow':
            tensorflow = TypeConverter._get_tensorflow()
            return tensorflow.convert_to_tensor(obj.__array__())
        # tensorflow -> jax
        elif TypeConverter._is_tensorflow_tensor(obj) and target_type == 'jax':
            jax = TypeConverter._get_jax()
            return jax.numpy.array(obj.numpy())
        # pandas -> xarray
        elif (TypeConverter._is_pandas_dataframe(obj) or TypeConverter._is_pandas_series(obj)) and target_type == 'xarray':
            xarray = TypeConverter._get_xarray()
            return xarray.DataArray(obj)
        # xarray -> pandas
        elif (TypeConverter._is_xarray_dataarray(obj) or TypeConverter._is_xarray_dataset(obj)) and target_type == 'pandas':
            return obj.to_pandas()
        
        # 转换为numpy数组作为中间格式
        if TypeConverter._is_numpy_array(obj):
            numpy_obj = obj
        elif TypeConverter._is_cupy_array(obj):
            cupy = TypeConverter._get_cupy()
            numpy_obj = cupy.asnumpy(obj)
        elif TypeConverter._is_scipy_sparse(obj):
            numpy_obj = obj.toarray()
        elif TypeConverter._is_pandas_dataframe(obj) or TypeConverter._is_pandas_series(obj):
            numpy_obj = obj.values
        elif TypeConverter._is_torch_tensor(obj):
            numpy_obj = obj.detach().cpu().numpy()
        elif TypeConverter._is_xarray_dataarray(obj):
            numpy_obj = obj.values
        elif TypeConverter._is_xarray_dataset(obj):
            if obj.data_vars:
                numpy_obj = next(iter(obj.data_vars.values())).values
            else:
                raise ValueError("Empty xarray Dataset")
        elif TypeConverter._is_jax_array(obj):
            numpy_obj = obj.__array__()
        elif TypeConverter._is_tensorflow_tensor(obj):
            numpy_obj = obj.numpy()
        else:
            # 尝试直接转换为numpy数组
            try:
                numpy = TypeConverter._get_numpy()
                if numpy is not None:
                    numpy_obj = numpy.array(obj)
                else:
                    raise ImportError("numpy is not available")
            except Exception as e:
                raise TypeError(f"Cannot convert {type(obj).__name__} to numpy array: {e}")
        
        # 从numpy数组转换为目标类型
        if target_type == 'numpy':
            return numpy_obj
        elif target_type == 'cupy':
            cupy = TypeConverter._get_cupy()
            return cupy.array(numpy_obj)
        elif target_type == 'scipy':
            scipy = TypeConverter._get_scipy()
            return scipy.sparse.csr_matrix(numpy_obj)
        elif target_type == 'pandas':
            return TypeConverter.numpy_to_pandas(numpy_obj)
        elif target_type == 'torch':
            torch = TypeConverter._get_torch()
            return torch.tensor(numpy_obj)
        elif target_type == 'xarray':
            return TypeConverter.numpy_to_xarray(numpy_obj)
        elif target_type == 'jax':
            jax = TypeConverter._get_jax()
            return jax.numpy.array(numpy_obj)
        elif target_type == 'tensorflow':
            tensorflow = TypeConverter._get_tensorflow()
            return tensorflow.constant(numpy_obj)
    
    @classmethod
    def _get_numpy(cls):
        """获取numpy模块
        
        Returns:
            numpy模块或None
        """
        if cls._numpy is None:
            try:
                import numpy
                cls._numpy = numpy
                cls._numpy_available = True
            except ImportError:
                cls._numpy_available = False
        return cls._numpy
    
    @classmethod
    def _get_cupy(cls):
        """获取cupy模块
        
        Returns:
            cupy模块或None
        """
        if cls._cupy is None:
            try:
                import cupy
                cls._cupy = cupy
                cls._cupy_available = True
            except ImportError:
                cls._cupy_available = False
        return cls._cupy
    
    @classmethod
    def _get_scipy(cls):
        """获取scipy模块
        
        Returns:
            scipy模块或None
        """
        if cls._scipy is None:
            try:
                import scipy
                cls._scipy = scipy
                cls._scipy_available = True
            except ImportError:
                cls._scipy_available = False
        return cls._scipy
    
    @classmethod
    def _get_pandas(cls):
        """获取pandas模块
        
        Returns:
            pandas模块或None
        """
        if cls._pandas is None:
            try:
                import pandas
                cls._pandas = pandas
                cls._pandas_available = True
            except ImportError:
                cls._pandas_available = False
        return cls._pandas
    
    @classmethod
    def _get_torch(cls):
        """获取torch模块
        
        Returns:
            torch模块或None
        """
        if cls._torch is None:
            try:
                import torch
                cls._torch = torch
                cls._torch_available = True
            except ImportError:
                cls._torch_available = False
        return cls._torch
    
    @classmethod
    def _get_xarray(cls):
        """获取xarray模块
        
        Returns:
            xarray模块或None
        """
        if cls._xarray is None:
            try:
                import xarray
                cls._xarray = xarray
                cls._xarray_available = True
            except ImportError:
                cls._xarray_available = False
        return cls._xarray
    
    @classmethod
    def _get_jax(cls):
        """获取jax模块
        
        Returns:
            jax模块或None
        """
        if cls._jax is None:
            try:
                import jax
                cls._jax = jax
                cls._jax_available = True
            except ImportError:
                cls._jax_available = False
        return cls._jax
    
    @classmethod
    def _get_tensorflow(cls):
        """获取tensorflow模块
        
        Returns:
            tensorflow模块或None
        """
        if cls._tensorflow is None:
            try:
                import tensorflow
                cls._tensorflow = tensorflow
                cls._tensorflow_available = True
            except ImportError:
                cls._tensorflow_available = False
        return cls._tensorflow
    
    @classmethod
    def _is_numpy_array(cls, obj):
        """检测是否为numpy数组
        
        Check if the object is a numpy array.
        
        Args:
            obj: 要检测的对象
            
        Returns:
            bool: 如果是numpy数组返回True，否则返回False
        """
        numpy = cls._get_numpy()
        return numpy is not None and isinstance(obj, numpy.ndarray)
    
    @classmethod
    def _is_cupy_array(cls, obj):
        """检测是否为cupy数组
        
        Check if the object is a cupy array.
        
        Args:
            obj: 要检测的对象
            
        Returns:
            bool: 如果是cupy数组返回True，否则返回False
        """
        cupy = cls._get_cupy()
        return cupy is not None and isinstance(obj, cupy.ndarray)
    
    @classmethod
    def _is_scipy_sparse(cls, obj):
        """检测是否为scipy稀疏矩阵
        
        Check if the object is a scipy sparse matrix.
        
        Args:
            obj: 要检测的对象
            
        Returns:
            bool: 如果是scipy稀疏矩阵返回True，否则返回False
        """
        scipy = cls._get_scipy()
        return scipy is not None and scipy.sparse.issparse(obj)
    
    @classmethod
    def _is_pandas_dataframe(cls, obj):
        """检测是否为pandas DataFrame
        
        Check if the object is a pandas DataFrame.
        
        Args:
            obj: 要检测的对象
            
        Returns:
            bool: 如果是pandas DataFrame返回True，否则返回False
        """
        pandas = cls._get_pandas()
        return pandas is not None and isinstance(obj, pandas.DataFrame)
    
    @classmethod
    def _is_pandas_series(cls, obj):
        """检测是否为pandas Series
        
        Check if the object is a pandas Series.
        
        Args:
            obj: 要检测的对象
            
        Returns:
            bool: 如果是pandas Series返回True，否则返回False
        """
        pandas = cls._get_pandas()
        return pandas is not None and isinstance(obj, pandas.Series)
    
    @classmethod
    def _is_torch_tensor(cls, obj):
        """检测是否为torch Tensor
        
        Check if the object is a torch Tensor.
        
        Args:
            obj: 要检测的对象
            
        Returns:
            bool: 如果是torch Tensor返回True，否则返回False
        """
        torch = cls._get_torch()
        return torch is not None and isinstance(obj, torch.Tensor)
    
    @classmethod
    def _is_xarray_dataarray(cls, obj):
        """检测是否为xarray DataArray
        
        Check if the object is an xarray DataArray.
        
        Args:
            obj: 要检测的对象
            
        Returns:
            bool: 如果是xarray DataArray返回True，否则返回False
        """
        xarray = cls._get_xarray()
        return xarray is not None and isinstance(obj, xarray.DataArray)
    
    @classmethod
    def _is_xarray_dataset(cls, obj):
        """检测是否为xarray Dataset
        
        Check if the object is an xarray Dataset.
        
        Args:
            obj: 要检测的对象
            
        Returns:
            bool: 如果是xarray Dataset返回True，否则返回False
        """
        xarray = cls._get_xarray()
        return xarray is not None and isinstance(obj, xarray.Dataset)
    
    @classmethod
    def _is_jax_array(cls, obj):
        """检测是否为jax Array
        
        Check if the object is a jax array.
        
        Args:
            obj: 要检测的对象
            
        Returns:
            bool: 如果是jax array返回True，否则返回False
        """
        jax = cls._get_jax()
        return jax is not None and isinstance(obj, jax.numpy.ndarray)
    
    @classmethod
    def _is_tensorflow_tensor(cls, obj):
        """检测是否为tensorflow Tensor
        
        Check if the object is a tensorflow Tensor.
        
        Args:
            obj: 要检测的对象
            
        Returns:
            bool: 如果是tensorflow Tensor返回True，否则返回False
        """
        tensorflow = cls._get_tensorflow()
        return tensorflow is not None and isinstance(obj, tensorflow.Tensor)
    
    @staticmethod
    @conversion_cache
    def to_list(obj):
        """转换为list
        
        Convert object to list.
        
        Args:
            obj: 要转换的对象
            
        Returns:
            list: 转换后的列表
            
        Raises:
            TypeError: 如果无法转换为列表
        """
        if isinstance(obj, list):
            return obj
        elif obj is None:
            return []
        elif isinstance(obj, (str, bytes)):
            return list(obj)
        elif isinstance(obj, dict):
            return list(obj.items())
        elif isinstance(obj, (int, float, bool)):
            return [obj]
        elif isinstance(obj, (tuple, set)):
            return list(obj)
        elif TypeConverter._is_numpy_array(obj):
            result = obj.tolist()
            # 处理0维数组（标量）
            if not isinstance(result, list):
                return [result]
            return result
        elif TypeConverter._is_cupy_array(obj):
            cupy = TypeConverter._get_cupy()
            result = cupy.asnumpy(obj).tolist()
            # 处理0维数组（标量）
            if not isinstance(result, list):
                return [result]
            return result
        elif TypeConverter._is_scipy_sparse(obj):
            result = obj.toarray().tolist()
            # 处理0维数组（标量）
            if not isinstance(result, list):
                return [result]
            return result
        elif TypeConverter._is_pandas_dataframe(obj):
            # 转换为嵌套列表，包含列名
            return [obj.columns.tolist()] + obj.values.tolist()
        elif TypeConverter._is_pandas_series(obj):
            # 转换为值列表
            return obj.tolist()
        elif TypeConverter._is_torch_tensor(obj):
            result = obj.tolist()
            # 处理0维张量（标量）
            if not isinstance(result, list):
                return [result]
            return result
        elif TypeConverter._is_xarray_dataarray(obj):
            result = obj.values.tolist()
            # 处理0维数组（标量）
            if not isinstance(result, list):
                return [result]
            return result
        elif TypeConverter._is_xarray_dataset(obj):
            # 转换为包含所有变量的字典列表
            return [{var: obj[var].values.tolist() for var in obj.data_vars}]
        elif TypeConverter._is_jax_array(obj):
            result = obj.tolist()
            # 处理0维数组（标量）
            if not isinstance(result, list):
                return [result]
            return result
        elif TypeConverter._is_tensorflow_tensor(obj):
            result = obj.numpy().tolist()
            # 处理0维张量（标量）
            if not isinstance(result, list):
                return [result]
            return result
        else:
            try:
                return original_list(obj)
            except (TypeError, ValueError):
                raise TypeError(f"Cannot convert {type(obj).__name__} to list")
    
    @staticmethod
    @conversion_cache
    def to_str(obj):
        """转换为str
        
        Convert object to string.
        
        Args:
            obj: 要转换的对象
            
        Returns:
            str: 转换后的字符串
            
        Raises:
            TypeError: 如果无法转换为字符串
        """
        if isinstance(obj, str):
            return obj
        elif obj is None:
            return ""
        elif isinstance(obj, bool):
            return str(obj).lower()
        elif isinstance(obj, (int, float, list, tuple, dict, set)):
            return original_str(obj)
        elif isinstance(obj, bytes):
            return str(obj, encoding='utf-8', errors='replace')
        elif TypeConverter._is_numpy_array(obj):
            return str(obj.tolist())
        elif TypeConverter._is_cupy_array(obj):
            cupy = TypeConverter._get_cupy()
            return str(cupy.asnumpy(obj).tolist())
        elif TypeConverter._is_scipy_sparse(obj):
            return str(obj.toarray().tolist())
        elif TypeConverter._is_pandas_dataframe(obj):
            return str(obj.to_dict('list'))
        elif TypeConverter._is_pandas_series(obj):
            return str(obj.to_list())
        elif TypeConverter._is_torch_tensor(obj):
            return str(obj.tolist())
        elif TypeConverter._is_xarray_dataarray(obj):
            return str(obj.values.tolist())
        elif TypeConverter._is_xarray_dataset(obj):
            return str({var: obj[var].values.tolist() for var in obj.data_vars})
        elif TypeConverter._is_jax_array(obj):
            return str(obj.tolist())
        elif TypeConverter._is_tensorflow_tensor(obj):
            return str(obj.numpy().tolist())
        else:
            try:
                return original_str(obj)
            except (TypeError, ValueError):
                raise TypeError(f"Cannot convert {type(obj).__name__} to str")
    
    @staticmethod
    @conversion_cache
    def to_int(obj):
        """转换为int
        
        Convert object to integer.
        
        Args:
            obj: 要转换的对象
            
        Returns:
            int: 转换后的整数
            
        Raises:
            TypeError: 如果无法转换为整数
            ValueError: 如果字符串无法转换为整数
        """
        if isinstance(obj, int):
            return obj
        elif obj is None:
            return 0
        elif isinstance(obj, bool):
            return int(obj)
        elif isinstance(obj, float):
            return int(obj)
        elif isinstance(obj, str):
            try:
                return int(obj.strip())
            except ValueError:
                raise ValueError(f"Cannot convert '{obj}' to int")
        elif isinstance(obj, bytes):
            try:
                return int(obj.decode('utf-8').strip())
            except (UnicodeDecodeError, ValueError):
                raise ValueError(f"Cannot convert bytes to int")
        elif isinstance(obj, (list, tuple, dict, set)):
            if len(obj) == 0:
                return 0
            elif len(obj) == 1:
                try:
                    if isinstance(obj, set):
                        return TypeConverter.to_int(next(iter(obj)))
                    return TypeConverter.to_int(obj[0])
                except (TypeError, ValueError):
                    raise TypeError(f"Cannot convert {type(obj).__name__} to int")
            else:
                raise TypeError(f"Cannot convert {type(obj).__name__} with length > 1 to int")
        elif TypeConverter._is_numpy_array(obj):
            if obj.size == 1:
                return int(obj.item())
            else:
                raise TypeError(f"Cannot convert numpy array with size > 1 to int")
        elif TypeConverter._is_cupy_array(obj):
            cupy = TypeConverter._get_cupy()
            if obj.size == 1:
                return int(cupy.asnumpy(obj).item())
            else:
                raise TypeError(f"Cannot convert cupy array with size > 1 to int")
        elif TypeConverter._is_scipy_sparse(obj):
            if obj.size == 1:
                return int(obj.toarray().item())
            else:
                raise TypeError(f"Cannot convert scipy sparse matrix with size > 1 to int")
        elif TypeConverter._is_pandas_dataframe(obj):
            if obj.size == 1:
                return int(obj.iloc[0, 0])
            else:
                raise TypeError(f"Cannot convert pandas DataFrame with size > 1 to int")
        elif TypeConverter._is_pandas_series(obj):
            if len(obj) == 1:
                return int(obj.iloc[0])
            else:
                raise TypeError(f"Cannot convert pandas Series with length > 1 to int")
        elif TypeConverter._is_torch_tensor(obj):
            if obj.numel() == 1:
                return int(obj.item())
            else:
                raise TypeError(f"Cannot convert torch Tensor with size > 1 to int")
        elif TypeConverter._is_xarray_dataarray(obj):
            if obj.size == 1:
                return int(obj.item())
            else:
                raise TypeError(f"Cannot convert xarray DataArray with size > 1 to int")
        elif TypeConverter._is_xarray_dataset(obj):
            raise TypeError(f"Cannot convert xarray Dataset to int")
        elif TypeConverter._is_jax_array(obj):
            if obj.size == 1:
                return int(obj.item())
            else:
                raise TypeError(f"Cannot convert jax array with size > 1 to int")
        elif TypeConverter._is_tensorflow_tensor(obj):
            if obj.shape == ():
                return int(obj.numpy())
            else:
                raise TypeError(f"Cannot convert tensorflow Tensor with size > 1 to int")
        else:
            try:
                return original_int(obj)
            except (TypeError, ValueError):
                raise TypeError(f"Cannot convert {type(obj).__name__} to int")
    
    @staticmethod
    @conversion_cache
    def to_float(obj):
        """转换为float
        
        Convert object to float.
        
        Args:
            obj: 要转换的对象
            
        Returns:
            float: 转换后的浮点数
            
        Raises:
            TypeError: 如果无法转换为浮点数
            ValueError: 如果字符串无法转换为浮点数
        """
        if isinstance(obj, float):
            return obj
        elif obj is None:
            return 0.0
        elif isinstance(obj, bool):
            return float(obj)
        elif isinstance(obj, int):
            return float(obj)
        elif isinstance(obj, str):
            try:
                return float(obj.strip())
            except ValueError:
                raise ValueError(f"Cannot convert '{obj}' to float")
        elif isinstance(obj, bytes):
            try:
                return float(obj.decode('utf-8').strip())
            except (UnicodeDecodeError, ValueError):
                raise ValueError(f"Cannot convert bytes to float")
        elif isinstance(obj, (list, tuple, dict, set)):
            if len(obj) == 0:
                return 0.0
            elif len(obj) == 1:
                try:
                    if isinstance(obj, set):
                        return TypeConverter.to_float(next(iter(obj)))
                    return TypeConverter.to_float(obj[0])
                except (TypeError, ValueError):
                    raise TypeError(f"Cannot convert {type(obj).__name__} to float")
            else:
                raise TypeError(f"Cannot convert {type(obj).__name__} with length > 1 to float")
        elif TypeConverter._is_numpy_array(obj):
            if obj.size == 1:
                return float(obj.item())
            else:
                raise TypeError(f"Cannot convert numpy array with size > 1 to float")
        elif TypeConverter._is_cupy_array(obj):
            cupy = TypeConverter._get_cupy()
            if obj.size == 1:
                return float(cupy.asnumpy(obj).item())
            else:
                raise TypeError(f"Cannot convert cupy array with size > 1 to float")
        elif TypeConverter._is_scipy_sparse(obj):
            if obj.size == 1:
                return float(obj.toarray().item())
            else:
                raise TypeError(f"Cannot convert scipy sparse matrix with size > 1 to float")
        elif TypeConverter._is_pandas_dataframe(obj):
            if obj.size == 1:
                return float(obj.iloc[0, 0])
            else:
                raise TypeError(f"Cannot convert pandas DataFrame with size > 1 to float")
        elif TypeConverter._is_pandas_series(obj):
            if len(obj) == 1:
                return float(obj.iloc[0])
            else:
                raise TypeError(f"Cannot convert pandas Series with length > 1 to float")
        elif TypeConverter._is_torch_tensor(obj):
            if obj.numel() == 1:
                return float(obj.item())
            else:
                raise TypeError(f"Cannot convert torch Tensor with size > 1 to float")
        elif TypeConverter._is_xarray_dataarray(obj):
            if obj.size == 1:
                return float(obj.item())
            else:
                raise TypeError(f"Cannot convert xarray DataArray with size > 1 to float")
        elif TypeConverter._is_xarray_dataset(obj):
            raise TypeError(f"Cannot convert xarray Dataset to float")
        elif TypeConverter._is_jax_array(obj):
            if obj.size == 1:
                return float(obj.item())
            else:
                raise TypeError(f"Cannot convert jax array with size > 1 to float")
        elif TypeConverter._is_tensorflow_tensor(obj):
            if obj.shape == ():
                return float(obj.numpy())
            else:
                raise TypeError(f"Cannot convert tensorflow Tensor with size > 1 to float")
        else:
            try:
                return original_float(obj)
            except (TypeError, ValueError):
                raise TypeError(f"Cannot convert {type(obj).__name__} to float")
    
    @staticmethod
    @conversion_cache
    def to_dict(obj):
        """转换为dict
        
        Convert object to dictionary.
        
        Args:
            obj: 要转换的对象
            
        Returns:
            dict: 转换后的字典
            
        Raises:
            TypeError: 如果无法转换为字典
        """
        if isinstance(obj, dict):
            return obj
        elif obj is None:
            return {}
        elif isinstance(obj, (list, tuple)):
            try:
                return dict(obj)
            except (ValueError, TypeError):
                # 如果是简单列表，转换为索引字典
                return {i: v for i, v in enumerate(obj)}
        elif isinstance(obj, (str, int, float, bool, set)):
            return {"value": obj}
        elif TypeConverter._is_numpy_array(obj):
            if obj.ndim == 1:
                return {i: v for i, v in enumerate(obj.tolist())}
            else:
                return {"shape": obj.shape, "dtype": str(obj.dtype), "data": obj.tolist()}
        elif TypeConverter._is_cupy_array(obj):
            cupy = TypeConverter._get_cupy()
            if obj.ndim == 1:
                return {i: v for i, v in enumerate(cupy.asnumpy(obj).tolist())}
            else:
                return {"shape": obj.shape, "dtype": str(obj.dtype), "data": cupy.asnumpy(obj).tolist()}
        elif TypeConverter._is_scipy_sparse(obj):
            array = obj.toarray()
            if array.ndim == 1:
                return {i: v for i, v in enumerate(array.tolist())}
            else:
                return {"shape": array.shape, "dtype": str(array.dtype), "data": array.tolist()}
        elif TypeConverter._is_pandas_dataframe(obj):
            return obj.to_dict('list')
        elif TypeConverter._is_pandas_series(obj):
            return obj.to_dict()
        elif TypeConverter._is_torch_tensor(obj):
            if obj.ndim == 1:
                return {i: v for i, v in enumerate(obj.tolist())}
            else:
                return {"shape": tuple(obj.shape), "data": obj.tolist()}
        elif TypeConverter._is_xarray_dataarray(obj):
            return {
                "shape": obj.shape,
                "dtype": str(obj.dtype),
                "data": obj.values.tolist(),
                "dims": obj.dims,
                "coords": {dim: obj.coords[dim].values.tolist() for dim in obj.dims}
            }
        elif TypeConverter._is_xarray_dataset(obj):
            return {
                var: {
                    "shape": obj[var].shape,
                    "dtype": str(obj[var].dtype),
                    "data": obj[var].values.tolist()
                }
                for var in obj.data_vars
            }
        elif TypeConverter._is_jax_array(obj):
            if obj.ndim == 1:
                return {i: v for i, v in enumerate(obj.tolist())}
            else:
                return {"shape": obj.shape, "dtype": str(obj.dtype), "data": obj.tolist()}
        elif TypeConverter._is_tensorflow_tensor(obj):
            if len(obj.shape) == 1:
                return {i: v for i, v in enumerate(obj.numpy().tolist())}
            else:
                return {"shape": tuple(obj.shape), "data": obj.numpy().tolist()}
        else:
            try:
                return original_dict(obj)
            except (TypeError, ValueError):
                raise TypeError(f"Cannot convert {type(obj).__name__} to dict")
    
    @staticmethod
    @conversion_cache
    def to_set(obj):
        """转换为set
        
        Convert object to set.
        
        Args:
            obj: 要转换的对象
            
        Returns:
            set: 转换后的集合
            
        Raises:
            TypeError: 如果无法转换为集合
        """
        if isinstance(obj, set):
            return obj
        elif obj is None:
            return set()
        elif isinstance(obj, (str, bytes)):
            return set(obj)
        elif isinstance(obj, (int, float, bool)):
            return {obj}
        elif isinstance(obj, (list, tuple, dict)):
            return set(obj)
        elif TypeConverter._is_numpy_array(obj):
            if obj.ndim == 1:
                return set(obj.tolist())
            else:
                raise TypeError(f"Cannot convert multi-dimensional numpy array to set")
        elif TypeConverter._is_cupy_array(obj):
            cupy = TypeConverter._get_cupy()
            if obj.ndim == 1:
                return set(cupy.asnumpy(obj).tolist())
            else:
                raise TypeError(f"Cannot convert multi-dimensional cupy array to set")
        elif TypeConverter._is_scipy_sparse(obj):
            array = obj.toarray()
            if array.ndim == 1:
                return set(array.tolist())
            else:
                raise TypeError(f"Cannot convert multi-dimensional scipy sparse matrix to set")
        elif TypeConverter._is_pandas_dataframe(obj):
            raise TypeError(f"Cannot convert pandas DataFrame to set")
        elif TypeConverter._is_pandas_series(obj):
            return set(obj.tolist())
        elif TypeConverter._is_torch_tensor(obj):
            if obj.ndim == 1:
                return set(obj.tolist())
            else:
                raise TypeError(f"Cannot convert multi-dimensional torch Tensor to set")
        elif TypeConverter._is_xarray_dataarray(obj):
            if obj.ndim == 1:
                return set(obj.values.tolist())
            else:
                raise TypeError(f"Cannot convert multi-dimensional xarray DataArray to set")
        elif TypeConverter._is_xarray_dataset(obj):
            raise TypeError(f"Cannot convert xarray Dataset to set")
        elif TypeConverter._is_jax_array(obj):
            if obj.ndim == 1:
                return set(obj.tolist())
            else:
                raise TypeError(f"Cannot convert multi-dimensional jax array to set")
        elif TypeConverter._is_tensorflow_tensor(obj):
            if len(obj.shape) == 1:
                return set(obj.numpy().tolist())
            else:
                raise TypeError(f"Cannot convert multi-dimensional tensorflow Tensor to set")
        else:
            try:
                return original_set(obj)
            except (TypeError, ValueError):
                raise TypeError(f"Cannot convert {type(obj).__name__} to set")
    
    @staticmethod
    @conversion_cache
    def to_tuple(obj):
        """转换为tuple
        
        Convert object to tuple.
        
        Args:
            obj: 要转换的对象
            
        Returns:
            tuple: 转换后的元组
            
        Raises:
            TypeError: 如果无法转换为元组
        """
        if isinstance(obj, tuple):
            return obj
        elif obj is None:
            return ()
        elif isinstance(obj, (str, bytes)):
            return tuple(obj)
        elif isinstance(obj, (int, float, bool)):
            return (obj,)
        elif isinstance(obj, (list, set, dict)):
            return tuple(obj)
        elif TypeConverter._is_numpy_array(obj):
            if obj.ndim == 1:
                return tuple(obj.tolist())
            else:
                return tuple(map(tuple, obj.tolist()))
        elif TypeConverter._is_cupy_array(obj):
            cupy = TypeConverter._get_cupy()
            if obj.ndim == 1:
                return tuple(cupy.asnumpy(obj).tolist())
            else:
                return tuple(map(tuple, cupy.asnumpy(obj).tolist()))
        elif TypeConverter._is_scipy_sparse(obj):
            array = obj.toarray()
            if array.ndim == 1:
                return tuple(array.tolist())
            else:
                return tuple(map(tuple, array.tolist()))
        elif TypeConverter._is_pandas_dataframe(obj):
            # 转换为包含列名和数据的嵌套元组
            return (tuple(obj.columns.tolist()),) + tuple(tuple(row) for row in obj.values.tolist())
        elif TypeConverter._is_pandas_series(obj):
            return tuple(obj.tolist())
        elif TypeConverter._is_torch_tensor(obj):
            if obj.ndim == 1:
                return tuple(obj.tolist())
            else:
                return tuple(map(tuple, obj.tolist()))
        elif TypeConverter._is_xarray_dataarray(obj):
            if obj.ndim == 1:
                return tuple(obj.values.tolist())
            else:
                return tuple(map(tuple, obj.values.tolist()))
        elif TypeConverter._is_xarray_dataset(obj):
            # 转换为包含所有变量的字典元组
            return (tuple({var: obj[var].values.tolist() for var in obj.data_vars}),)
        elif TypeConverter._is_jax_array(obj):
            if obj.ndim == 1:
                return tuple(obj.tolist())
            else:
                return tuple(map(tuple, obj.tolist()))
        elif TypeConverter._is_tensorflow_tensor(obj):
            if len(obj.shape) == 1:
                return tuple(obj.numpy().tolist())
            else:
                return tuple(map(tuple, obj.numpy().tolist()))
        else:
            try:
                return original_tuple(obj)
            except (TypeError, ValueError):
                raise TypeError(f"Cannot convert {type(obj).__name__} to tuple")

# 重写内置转换函数
def e2e_list(obj=None):
    """转换为list
    
    Convert object to list, with support for all standard types and third-party library types.
    
    Args:
        obj: 要转换的对象，默认为None
        
    Returns:
        list: 转换后的列表
    """
    if obj is None:
        return []
    return TypeConverter.to_list(obj)

def e2e_str(obj=""):
    """转换为str
    
    Convert object to string, with support for all standard types and third-party library types.
    
    Args:
        obj: 要转换的对象，默认为空字符串
        
    Returns:
        str: 转换后的字符串
    """
    return TypeConverter.to_str(obj)

def e2e_int(obj=0):
    """转换为int
    
    Convert object to integer, with support for all standard types and third-party library types.
    
    Args:
        obj: 要转换的对象，默认为0
        
    Returns:
        int: 转换后的整数
    """
    return TypeConverter.to_int(obj)

def e2e_float(obj=0.0):
    """转换为float
    
    Convert object to float, with support for all standard types and third-party library types.
    
    Args:
        obj: 要转换的对象，默认为0.0
        
    Returns:
        float: 转换后的浮点数
    """
    return TypeConverter.to_float(obj)

def e2e_dict(obj=None):
    """转换为dict
    
    Convert object to dictionary, with support for all standard types and third-party library types.
    
    Args:
        obj: 要转换的对象，默认为None
        
    Returns:
        dict: 转换后的字典
    """
    if obj is None:
        return {}
    return TypeConverter.to_dict(obj)

def e2e_set(obj=None):
    """转换为set
    
    Convert object to set, with support for all standard types and third-party library types.
    
    Args:
        obj: 要转换的对象，默认为None
        
    Returns:
        set: 转换后的集合
    """
    if obj is None:
        return set()
    return TypeConverter.to_set(obj)

def e2e_tuple(obj=()):
    """转换为tuple
    
    Convert object to tuple, with support for all standard types and third-party library types.
    
    Args:
        obj: 要转换的对象，默认为空元组
        
    Returns:
        tuple: 转换后的元组
    """
    return TypeConverter.to_tuple(obj)
