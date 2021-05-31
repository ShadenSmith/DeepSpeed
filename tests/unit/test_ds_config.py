import pytest
import os
import json
from deepspeed.runtime import config as ds_config

import deepspeed.config as config


def test_only_required_fields(tmpdir):
    '''Ensure that config containing only the required fields is accepted. '''
    cfg_json = tmpdir.mkdir('ds_config_unit_test').join('minimal.json')

    with open(cfg_json, 'w') as f:
        required_fields = {'train_batch_size': 64}
        json.dump(required_fields, f)

    run_cfg = ds_config.DeepSpeedConfig(cfg_json)
    assert run_cfg is not None
    assert run_cfg.train_batch_size == 64
    assert run_cfg.train_micro_batch_size_per_gpu == 64
    assert run_cfg.gradient_accumulation_steps == 1


def test_config_duplicate_key(tmpdir):
    config_dict = '''
    {
        "train_batch_size": 24,
        "train_batch_size": 24,
    }
    '''
    config_path = os.path.join(tmpdir, 'temp_config.json')

    with open(config_path, 'w') as jf:
        jf.write("%s" % config_dict)

    with pytest.raises(ValueError):
        run_cfg = ds_config.DeepSpeedConfig(config_path)


class MyConfig(config.Config):
    verbose: bool
    name: str = 'Beluga'


def test_base_config():

    c = MyConfig(verbose=True)
    assert c.verbose == True
    assert c['verbose'] == True
    assert c.name == 'Beluga'

    # test iterators
    assert list(c.keys()) == ['verbose', 'name']
    assert list(c.values()) == [True, 'Beluga']
    for x, y in zip(c.items(), [('verbose', True), ('name', 'Beluga')]):
        assert x == y


def test_config_register():
    c = MyConfig(verbose=False)

    with pytest.raises(ValueError):
        c.register_arg('verbose', True)

    c.register_arg('level', 11)
    assert c.level == 11
    assert 'level' in c.keys()


@pytest.mark.skip('Not impemented yet')
def test_batch_config():
    config_dict = {
        'train_batch_size': 4,
        'micro_batch_size': 2,
    }

    c = config.BatchConfig(**config_dict)
    print()
    print(c)
    assert c['train_batch_size'] == 2
    assert c.gradient_accumulation_steps == 2


class AliasedConfig(config.Config):
    age: int = 1
    age2: int = config.alias('age', deprecated=True)
    name: str = 'tygra'


def test_alias():
    c = AliasedConfig(age=1)

    assert c.age == c.age2

    c.age = 11
    assert c.age == c.age2

    c.age2 = 1138
    assert c.age == c.age2

    c2 = AliasedConfig.from_dict(c.as_dict())
    c2.age2 = 111111
    assert c2.age == c2.age2

    # legacy dicts?
    legacy = {'age2': 3, 'name': 'Panthro'}
    c = AliasedConfig.from_dict(legacy)
    assert c.age == 3
    assert c.age2 == 3
    assert c.name == 'Panthro'


def test_config_str():
    c = MyConfig(verbose=False)
    print()
    print(c)
