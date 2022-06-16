import os
from typing import Any, Iterable, Optional

import numpy as np
import random
import scipy.signal
import torch

# ******************************************************
# ******************************************************
# ******************** math utils **********************


def combined_shape(length, shape=None):
    if shape is None:
        return (length, )
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def discount_cumsum(x, discount):
    r"""
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        numpy 1d vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


# ******************************************************
# ******************************************************
# ******************** torch utils *********************
def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    # torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def export_device_env_variable(device: str, id=0):
    r'''
    Export a local env variable to specify the device for all tensors.
    Only call this function once in the beginning of a job script.

    @param device: should be "gpu" or "cpu"
    @param id: gpu id
    '''
    if device.lower() == "gpu":
        if torch.cuda.is_available():
            os.environ["MODEL_DEVICE"] = 'cuda:' + str(id)
            return True
    os.environ["MODEL_DEVICE"] = 'cpu'


def get_torch_device():
    '''
    Return the torch.device class based on environment variable 'MODEL_DEVICE'.
    '''
    device_name = os.environ.get("MODEL_DEVICE")
    try:
        return torch.device(device_name)
    except:
        raise ValueError(
            f"'MODEL_DEVICE' env variable has not been specified or is not specified correctly, please export this env variable first! Current 'MODEL_DEVICE' env variable is {device_name}"
        )


def get_device_name():
    '''
    Return the environment variable 'MODEL_DEVICE'
    '''
    return os.environ.get("MODEL_DEVICE")


def to_tensor(item: Any,
              dtype: torch.dtype = torch.float32,
              device: Optional[torch.device] = None,
              ignore_keys: list = [],
              transform_scalar: bool = True,
              squeeze=False) -> torch.Tensor:
    r"""
    Overview:
        Change `numpy.ndarray`, sequence of scalars to torch.Tensor, and keep other data types unchanged.
    Arguments:
        - item (:obj:`Any`): the item to be changed
        - dtype (:obj:`type`): the type of wanted tensor
    Returns:
        - item (:obj:`torch.Tensor`): the change tensor
    .. note:

        Now supports item type: :obj:`dict`, :obj:`list`, :obj:`tuple` and :obj:`None`
    """
    device = get_torch_device() if device is None else device

    def squeeze_tensor(d):
        data = torch.tensor(d, dtype=dtype, device=device)
        if squeeze:
            return torch.squeeze(data)
        return data

    if isinstance(item, dict):
        new_data = {}
        for k, v in item.items():
            if k in ignore_keys:
                new_data[k] = v
            else:
                new_data[k] = to_tensor(v,
                                        dtype,
                                        device,
                                        ignore_keys,
                                        transform_scalar,
                                        squeeze=squeeze)
        return new_data

    elif isinstance(item, list) or isinstance(item, tuple):
        if len(item) == 0:
            return None
        return squeeze_tensor(item)

    elif isinstance(item, np.ndarray):
        return squeeze_tensor(item)

    elif isinstance(item, bool) or isinstance(item, str):
        return item

    elif np.isscalar(item):
        if transform_scalar:
            return torch.as_tensor(item, device=device).to(dtype)
        else:
            return item

    elif item is None:
        return None

    elif isinstance(item, torch.Tensor):
        return item.to(dtype)
    else:
        raise TypeError("not support item type: {}".format(type(item)))


def to_ndarray(item: Any, dtype: np.dtype = None) -> np.ndarray:
    r"""
    Overview:
        Change `torch.Tensor`, sequence of scalars to ndarray, and keep other data types unchanged.
    Arguments:
        - item (:obj:`object`): the item to be changed
        - dtype (:obj:`type`): the type of wanted ndarray
    Returns:
        - item (:obj:`object`): the changed ndarray
    .. note:

        Now supports item type: :obj:`torch.Tensor`,  :obj:`dict`, :obj:`list`, :obj:`tuple` and :obj:`None`
    """
    def transform(d):
        if dtype is None:
            return np.array(d)
        else:
            return np.array(d, dtype=dtype)

    if isinstance(item, dict):
        new_data = {}
        for k, v in item.items():
            new_data[k] = to_ndarray(v, dtype)
        return new_data
    elif isinstance(item, list) or isinstance(item, tuple):
        if len(item) == 0:
            return None
        elif hasattr(item, '_fields'):  # namedtuple
            return type(item)(*[to_ndarray(t, dtype) for t in item])
        else:
            new_data = []
            for t in item:
                new_data.append(to_ndarray(t, dtype))
            return new_data
    elif isinstance(item, torch.Tensor):
        if item.device != 'cpu':
            item = item.detach().cpu()
        if dtype is None:
            return item.numpy()
        else:
            return item.numpy().astype(dtype)
    elif isinstance(item, np.ndarray):
        if dtype is None:
            return item
        else:
            return item.astype(dtype)
    elif isinstance(item, bool) or isinstance(item, str):
        return item
    elif np.isscalar(item):
        return np.array(item)
    elif item is None:
        return None
    else:
        raise TypeError("not support item type: {}".format(type(item)))


def to_device(item: Any, device: str = None, ignore_keys: list = []) -> Any:
    r"""
    Overview:
        Transfer data to certain device
    Arguments:
        - item (:obj:`Any`): the item to be transferred
        - device (:obj:`str`): the device wanted, could be get_torch_device()
        - ignore_keys (:obj:`list`): the keys to be ignored in transfer, defalut set to empty
    Returns:
        - item (:obj:`Any`): the transferred item
    .. note:

        Now supports item type: :obj:`torch.nn.Module`, :obj:`torch.Tensor`, \
            :obj:`dict`, :obj:`np.ndarray`, :obj:`str` and :obj:`None`.

    """
    if device is None:
        device = get_device_name()
    if isinstance(item, torch.nn.Module):
        return item.to(device)
    elif isinstance(item, torch.Tensor):
        return item.to(device)
    elif isinstance(item, dict):
        new_item = {}
        for k in item.keys():
            if k in ignore_keys:
                new_item[k] = item[k]
            else:
                new_item[k] = to_device(item[k], device)
        return new_item
    elif isinstance(item, np.ndarray) or isinstance(item, np.bool_):
        return item
    elif item is None or isinstance(item, str):
        return item
    elif isinstance(item, list):
        return [to_device(k, device) for k in item]
    elif isinstance(item, tuple):
        return tuple([to_device(k, device) for k in item])
    else:
        raise TypeError("not support item type: {}".format(type(item)))


def to_dtype(item: Any, dtype: type) -> Any:
    r"""
    Overview:
        Change data to certain dtype
    Arguments:
        - item (:obj:`Any`): the item to be dtype changed
        - dtype (:obj:`type`): the type wanted
    Returns:
        - item (:obj:`object`): the dtype changed item
    .. note:

        Now supports item type: :obj:`torch.Tensor`, :obj:`dict`
    """
    if isinstance(item, torch.Tensor):
        return item.to(dtype=dtype)
    elif isinstance(item, dict):
        return {k: to_dtype(item[k], dtype) for k in item.keys()}
    else:
        raise TypeError("not support item type: {}".format(type(item)))


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def CUDA(var):
    return var.cuda() if torch.cuda.is_available() else var


def CPU(var):
    return var.cpu().detach()