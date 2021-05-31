import torch

from .base import *


class BatchConfig(Config):
    r"""Configure parameters related to batch sizes.

    .. math::

        \begin{eqnarray}
            \text{train_batch_size} & = & \text{micro_batch_size} \\
                                    & & \times \text{gradient_accumulation_steps} \times \text{data parallelism}
        \end{eqnarray}

    """

    train_batch_size: int = None
    """ The effective training batch size.

    This is the number of data samples that leads to one step of model
    update. :attr:`train_batch_size` is aggregated by the batch size that a
    single GPU processes in one forward/backward pass (a.k.a.,
    :attr:`train_step_batch_size`), the gradient accumulation steps (a.k.a.,
    :attr:`gradient_accumulation_steps`), and the number of GPUs.
    """

    micro_batch_size: int = None
    """The batch size to be processed per device each forward/backward step.

    When specified, ``gradient_accumulation_steps`` is automatically
    calculated using ``train_batch_size`` and the number of devices. Should
    not be concurrently specified with ``gradient_accumulation_steps``.
    """

    gradient_accumulation_steps: int = None
    """ The number of training steps to accumulate gradients before averaging
    and applying them.

    This feature is sometimes useful to improve scalability
    since it results in less frequent communication of gradients between
    steps. Another impact of this feature is the ability to train with larger
    batch sizes per GPU. When specified, ``train_step_batch_size`` is
    automatically calculated using ``train_batch_size`` and number of GPUs.
    Should not be concurrently specified with ``train_batch_size``.
    """

    train_micro_batch_size_per_gpu: int = alias('micro_batch_size')
    """Alias of :attr:`micro_batch_size`.

    .. note::

        Configuration ``train_micro_batch_size_per_gpu`` is deprecated and will
        be removed in future releases. See :attr:`micro_batch_size` instead.
    """
    def _resolve(self):
        """Complete batch configuration so long as two are provided. """
        batch = self.train_batch_size
        mb = self.train_micro_batch_size_per_gpu
        gas = self.gradient_accumulation_steps

        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
        else:
            world_size = 1
        self.world_size = world_size

        # All values are provided, nothing needs to be set
        if all([batch, mb, gas]):
            return

        #global_accumulation_steps needs to be set
        elif batch is not None and \
            mb is not None:
            gas = batch // mb
            gas //= world_size
            self.gradient_accumulation_steps = gas

        #micro_batch_per_gpu needs to be set
        elif batch is not None and \
            gas is not None:
            mb = batch // world_size
            mb //= gas
            self.train_micro_batch_size_per_gpu = mb

        #train_batch_size needs to be set
        elif mb is not None and \
            gas is not None:
            batch = mb * gas
            batch *= world_size
            self.train_batch_size = batch

        #gradient_accumulation_steps and micro_batch_per_gpus is set
        elif batch is not None:
            self.gradient_accumulation_steps = 1
            self.train_micro_batch_size_per_gpu = batch // world_size

        #train_batch_size and gradient_accumulation_step is set
        elif mb is not None:
            self.train_batch_size = mb * world_size
            self.gradient_accumulation_steps = 1

    def is_valid(self):
        self.resolve()

        batch = self.train_batch_size
        mb = self.train_micro_batch_size_per_gpu
        gas = self.gradient_accumulation_steps

        if batch is None or batch <= 0:
            raise ConfigError(f'train_batch_size: {batch} must be greater than 0.')

        if mb is None or mb <= 0:
            raise ConfigError(
                f'train_micro_batch_size_per_gpu: {mb} must be greater than 0.')

        if gas is None or gas <= 0:
            raise ConfigError(
                f'gradient_accumulation_steps: {gas} must be greater than 0.')

        if batch != (mb * gas * self.world_size):
            raise ConfigError(
                f'Check batch related parameters. train_batch_size is not equal'
                f' to micro_batch_per_gpu * gradient_acc_step * world_size'
                f'{batch} != {mb} * {gas} * {self.world_size}')

        return True


class FP16Config(Config):
    """ FP16 configuration. """

    #: Enable/disable FP16
    enabled: bool = False

    #: Gradient clipping
    clip: float = 1.0


class TrainingConfig(Config):
    """Top-level configuration for all aspects of training with DeepSpeed."""

    #: Batch configuration, see :class:`BatchConfig`
    #batch: BatchConfig = None
    batch = None

    #: FP16 training, see :class:`FP16Config`
    fp16 = None
