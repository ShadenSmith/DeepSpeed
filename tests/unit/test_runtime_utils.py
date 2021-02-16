import pytest
import torch

import deepspeed.runtime.utils as ds_utils


def test_call_to_str():
    c2s = ds_utils.call_to_str

    assert c2s('int') == 'int()'
    assert c2s('int', 3) == 'int(3)'
    assert c2s('int', 3, 'jeff') == 'int(3, \'jeff\')'

    assert c2s('hello', val=3) == 'hello(val=3)'
    assert c2s('hello', 1138, val=3) == 'hello(1138, val=3)'


def test_one_or_many():
    @ds_utils.one_or_many
    def _double(x):
        return x * 2

    # test basic mechanics
    assert _double(1) == 2
    assert _double([1]) == [2]
    assert _double([1, 2]) == [2, 4]
    assert _double((1, 2)) == (2, 4)

    @ds_utils.one_or_many
    def _add(x, y):
        return x + y

    assert _add(1, 1) == 2
    assert _add(1, y=1) == 2
    assert _add([1, 2], y=1) == [2, 3]
    assert _add((1, 2), y=1) == (2, 3)

    # Test list of lists. Should get a list of sums.
    @ds_utils.one_or_many
    def _sum(losses):
        return sum(losses)

    ls = [[torch.LongTensor([x * y]) for x in range(1, 4)] for y in range(3)]
    summed = _sum(ls)
    assert len(summed) == len(ls)
    for idx, result in enumerate(summed):
        assert result.item() == 6 * idx
