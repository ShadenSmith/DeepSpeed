import json
import dataclasses
import typing
from typing import Generator, Iterator, Tuple, Type, Any


class ConfigError(Exception):
    """Errors related to DeepSpeed configuration. """
    pass


class ConfigArg:
    def __init__(self, default=None, value=None):
        self.default = default
        if value is not None:
            self.value = value
        else:
            self.value = default

    def is_valid(self):
        return True

    def __repr__(self):
        return str(self.value)


class RequiredArg(ConfigArg):
    def __init__(self):
        super().__init__(default=None)

    def is_valid(self):
        # Ensure the required argument is provided.
        if self.value is None:
            return False
        return super().is_valid()


class SubConfig(ConfigArg):
    def __init__(self, config):
        if not isinstance(config, Config):
            raise TypeError(f'Expecting type Config, got {type(config)}')
        super().__init__(value=config)

    def is_valid(self):
        return self.value.is_valid()


class MetaConfig(type):
    """A metaclass that injects `dataclasses.dataclass` behavior into a class.

    We implement as a metaclass here to retain dataclass behavior without a
    `@dataclass` decorator for each subclass.
    """
    def __new__(cls, name, bases, dct):
        return dataclasses.dataclass(super().__new__(cls, name, bases, dct))


# Config typing for factory methods
T = typing.TypeVar('T', bound='Config')


class Config(metaclass=MetaConfig):
    """Base class for DeepSpeed configurations.

    ``Config`` is a dataclass with subclassing. Configurations should be subclassed
    to group arguments by topic.

    .. code-block:: python

        class MyConfig(Config):
            verbose: bool
            name: str = 'Beluga' # default value


    Configurations are initialized from dictionaries or keyword arguments:

        >>> myconf = {'verbose' : True}
        >>> c = MyConfig(**myconf)
        >>> c = MyConfig(verbose=True)
        >>> c['verbose']
        True
        >>> c.name
        'Beluga'
    """
    def __post_init__(self):
        """Collect all of the class-specified config arg names into a list."""
        self._arg_names = [f.name for f in dataclasses.fields(self)]

    def register_arg(self, name: str, value: Any):
        """Add an argument to an existing config.

        Args:
            name (str): The name of the config arg to add.
            value (Any): The value of the config arg.

        Raises:
            ValueError: if `name` is already a config arg.
        """
        if name in self._arg_names:
            raise ValueError(f'config arg {name} already registered with {type(self)}')

        self._arg_names.append(name)
        setattr(self, name, value)

    def items(self) -> Iterator[Tuple[str, Any]]:
        """Yields tuples of config arguments in the form (`name`, `value`).

        This is equivalent to `dict.items`.

        .. note::

            Only arguments specified in the class or added via :meth:`register_arg`
            will be included in the generator.

        Yields:
            Tuple[str, Any]: the configuration's arguments.
        """
        for name in self._arg_names:
            value = getattr(self, name)
            yield name, value

    def keys(self) -> Iterator[str]:
        """Yields the names of config arguments.

        Yields:
            Iterator[str]: The names of configuration arguments.
        """
        for k, _ in self.items():
            yield k

    def values(self) -> Iterator[Any]:
        """Yields the values of config arguments.

        Yields:
            Iterator[Any]: The configured argument values.
        """
        for _, v in self.items():
            yield v

    def as_dict(self) -> dict:
        """Return a copy of the config represented as a dictionary.

        .. code-block:: python

            class MyConfig(Config):
                verbose: bool
                name: str = 'Beluga'

            >>> MyConfig(verbose=True).as_dict()
            {'verbose': True, 'name': 'Beluga'}

        Returns:
            dict: the config dictionary
        """
        return dataclasses.asdict(self)

    def resolve(self):
        """Infer any missing arguments, if applicable.

        This is useful for configs such as :class:`BatchConfig` in only a
        subset of arguments are required to complete a valid config.
        """

        # Walk the tree of subconfigs and also resolve().
        for arg in self._args:
            if isinstance(arg, SubConfig):
                arg.resolve()

    @classmethod
    def from_dict(cls: Type[T], config: dict) -> T:
        """Construct a config from a dictionary.

        Equivalent to:

        .. code-block:: python

            config = {'verbose' : True, 'name' : 'Beluga'}
            c = MyConfig(**config)

        Args:
            cls (`Config`): The config class to construct.
            config (dict): A path to the JSON file to parse.

        Returns:
            The constructed config.
        """
        return cls(**config)

    @classmethod
    def from_json(cls: Type[T], json_path: str) -> T:
        """Parse a JSON file and return a configuration.

        Args:
            cls (`Config`): The config class to construct.
            json_path (str): A path to the JSON file to parse.

        Returns:
            The constructed config.
        """
        with open(json_path, 'r') as fin:
            config_dict = json.load(fin)
        return cls.from_dict(**config_dict)

    def is_valid(self) -> bool:
        """Resolve any missing configurations and determine in the configuration is valid.

        Returns:
            bool: Whether the config and all sub-configs are valid.
        """
        self.resolve()
        return all(arg.is_valid() for arg in self._args.values())

    def __str__(self) -> str:
        return self.dot_str()

    def dot_str(self, depth: int = 0, dots_width: int = 50) -> str:
        indent_width = 4
        indent = ' ' * indent_width
        lines = []
        lines.append(f'{indent * depth}{self.__class__.__name__} = {{')

        for key, val in self.items():
            # Recursive configurations
            if isinstance(val, SubConfig):
                config = val.value
                lines.append(config.dot_str(depth=depth + 1))
                continue

            dots = '.' * (dots_width - len(key) - (depth * indent_width))
            lines.append(f'{indent * (depth+1)}{key} {dots} {repr(val)}')
        lines.append(f'{indent * depth}}}')
        return '\n'.join(lines)

    def __getitem__(self, name: str) -> Any:
        """Transparently support dict-style accesses."""
        return getattr(self, name)
