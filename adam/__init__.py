import numpy as np
import torch
import os
import requests
import traceback

__all__ = ["with_time", "to_signals", "standardize"]

# this should be an array a column for each signal  and any number of rows
Signals = np.ndarray

cache_dir = os.path.expanduser("~/.adam/jobcache")


class Notifier(object):
    def __enter__(self):
        return None

    def __exit__(self, type, value, tb):
        traceback.print_tb(tb)
        requests.post(
            "https://maker.ifttt.com/trigger/notification/with/key/dbDmGmj4zjYdGgX5HEW1EW",
            json={"value1": "error"},
        )
        return True


def is_signals(val) -> bool:
    return (
        isinstance(val, torch.Tensor)
        and len(val.shape) == 2
        and val.shape[0] > 0
        and val.shape[1] > 0
    )


def assert_is_signals(val):
    assert is_signals(val)


def to_signals(signal: torch.Tensor) -> torch.Tensor:
    """
    :return: an array with one column and n rows where n=len(signal)
    """
    num_axes = len(signal.shape)
    if num_axes not in range(1, 3):
        raise Exception(
            "An np.ndarray representing a single signal should have 1 or 2 axes but signal had {num_axes} axes"
        )
    elif num_axes == 1:
        return signal.reshape((-1, 1))
    elif num_axes == 2:
        return signal


def with_time_if_necessary(signals):
    return (
        signals
        if (len(signals.shape) > 1 and signals.shape[1] == 2)
        else with_time(signals)
    )


def range_std(arr: torch.Tensor):
    return arr / (torch.max(arr) - torch.min(arr))


def standardize(arr: np.ndarray):
    """
    Standardizes along the sepcified axis. Axis is 0 by default and in that case,
    each column will be standardized. 
    """
    means = np.mean(arr, axis=0)
    return (arr - means) / np.std(arr, axis=0)


# def with_time(arr: np.ndarray):
#     time = np.arange(len(arr))
#     return np.insert(to_signals(arr), 0, time / np.mean(np.abs(np.diff(time))), axis=1)


def with_time(arr: torch.Tensor):
    arr = to_signals(arr)
    new_arr = torch.zeros((arr.shape[0], 1 + arr.shape[1]))
    time = torch.arange(len(arr)).float()
    new_arr[:, 0] = time / time[-1]
    new_arr[:, 1:] = arr
    return new_arr
