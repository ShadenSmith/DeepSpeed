import json
import torch

#from .base import *

from dataclasses import dataclass, field


@dataclass
class Config:
    def __getitem__(self, name):
        """Transparently support dict-style accesses."""
        return getattr(self, name)

    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)

    @classmethod
    def from_json(cls, json_path):
        with open(json_path, 'r') as fin:
            config_dict = json.load(fin)
        return cls.from_dict(**config_dict)


class ArgAlias:
    def __init__(self, argname):
        self.argname = argname
        self.fget = lambda obj: getattr(obj, self.argname)
        self.fset = lambda obj, val: setattr(obj, self.argname, val)
        self.__doc__ = f'Alias for :attr:`{argname}`.'

    def __get__(self, obj):
        return self.fget(obj)

    def __set__(self, obj, value):
        self.fset(obj, value)

    def getter(self, fget):
        return type(self)(fget, self.fset, self.fdel, self.__doc__)

    def setter(self, fset):
        return type(self)(self.fget, fset, self.fdel, self.__doc__)


@dataclass
class BatchConfig(Config):
    r"""Configure parameters related to bath sizes.

    .. math::

        \begin{eqnarray}
            \text{train_batch_size} & = & \text{train_micro_batch_size_per_gpu} \\
                                    & & \times \text{gradient_accumulation_steps} \times \text{data parallelism}
        \end{eqnarray}
    """

    train_batch_size: int = None
    """The effective training batch size.
    This is the number of data samples that leads to one step of model
    update. :attr:`train_batch_size` is aggregated by the batch size that a
    single GPU processes in one forward/backward pass (a.k.a.,
    :attr:`train_step_batch_size`), the gradient accumulation steps (a.k.a.,
    :attr:`gradient_accumulation_steps`), and the number of GPUs.
    """

    micro_batch_size: int = 1
    """The batch size to be processed per device each forward/backward step.

    When specified, ``gradient_accumulation_steps`` is automatically
    calculated using ``train_batch_size`` and the number of devices. Should
    not be concurrently specified with ``gradient_accumulation_steps``.
    """

    gradient_accumulation_steps: int = 1
    """The number of training steps to accumulate gradients before averaging
    and applying them.

    This feature is sometimes useful to improve scalability
    since it results in less frequent communication of gradients between
    steps. Another impact of this feature is the ability to train with larger
    batch sizes per GPU. When specified, ``train_step_batch_size`` is
    automatically calculated using ``train_batch_size`` and number of GPUs.
    Should not be concurrently specified with ``train_step_batch_size``.
    """

    train_micro_batch_size_per_gpu: int = ArgAlias('micro_batch_size')


def test_dc():
    c = BatchConfig(train_batch_size=2)

    print()
    print(c)
    assert c.train_batch_size == 2
    assert c['train_batch_size'] == 2
    assert c.gradient_accumulation_steps == 1
