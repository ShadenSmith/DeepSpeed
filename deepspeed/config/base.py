import json
import hjson
import dataclasses
import typing
from typing import Iterator, Union, Tuple, Type, Any


class ConfigError(Exception):
    """Errors related to DeepSpeed configuration. """
    pass


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

        for name, arg in self.items():
            if isinstance(arg, AliasSpec):
                setattr(self.__class__, name, arg.build(root_config=self))
        '''
        # Build alias arguments
        aliases = filter(is_alias, self.items())
        print()
        from types import MethodType
        for alias_name, alias in aliases:
            #prop = property(lambda obj: getattr(obj, alias.argname))
            #prop = MethodType(lambda self: self[alias.argname], self)
            #setattr(self, alias_name, prop)
            pass
        '''

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
        """Infer any missing arguments in this config and subconfigs, if applicable.

        This is useful for configs such as :class:`BatchConfig` in only a
        subset of arguments are required to complete a valid config.

        .. note::

            This method is called automatically by :meth:`deepspeed.initialize`, but can
            be optionally used before.
        """

        self._resolve()

        # Walk the tree of subconfigs and also resolve().
        for key, arg in self.items():
            if isinstance(arg, Config):
                arg.resolve()

    def _resolve(self):
        """Implementation to infer any missing arguments for an individual config.

        .. note::

            Subclasses of :class:`Config` should implement this method to handle arguments
            whose values can be inferred at :meth:`resolve` time.
        """
        pass

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
    def from_json(cls: Type[T], path: str) -> T:
        """Parse a JSON file and return a configuration.

        Args:
            cls (`Config`): The config class to construct.
            path (str): A path to the JSON file to parse.

        Returns:
            The constructed config.
        """
        with open(path, 'r') as fin:
            config_dict = json.load(fin)
        return cls.from_dict(**config_dict)

    @classmethod
    def from_hjson(cls: Type[T], path: str) -> T:
        """Parse an Hjson file and return a configuration.

        Args:
            cls (`Config`): The config class to construct.
            path (str): A path to the JSON file to parse.

        Returns:
            The constructed config.
        """
        with open(path, 'r') as fin:
            config_dict = hjson.load(fin)
        return cls.from_dict(**config_dict)

    def is_valid(self) -> bool:
        """Resolve any missing configurations and determine in the configuration is valid.

        Returns:
            bool: Whether the config and all sub-configs are valid.
        """
        self.resolve()
        return all(arg.is_valid() for arg in self.values())

    def __str__(self) -> str:
        return self.dot_str()

    def dot_str(self, depth: int = 0, dots_width: int = 50) -> str:
        indent_width = 4
        indent = ' ' * indent_width
        lines = []
        lines.append(f'{indent * depth}{self.__class__.__name__} = {{')

        for key, val in self.items():
            # Recursive configurations
            if isinstance(val, Config):
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


class ArgAlias:
    def __init__(self, name, deprecated: bool = True):
        self.name = name

    def __get__(self, inst, cls):
        if inst is None:
            # class attribute
            return self
        return getattr(inst, self.name)

    def __set__(self, inst, value):
        setattr(inst, self.name, value)

    def __delete__(self, inst):
        delattr(inst, self.name)

    def __repr__(self) -> str:
        return f"(alias to {self.name})'"


class AliasSpec:
    def __init__(self, alias_to: str, deprecated: bool):
        self.alias_to = alias_to
        self.deprecated = deprecated

    def build(self, root_config: Config) -> ArgAlias:
        if not hasattr(root_config, self.alias_to):
            raise RuntimeError(f'Config {type(root_config)} has no attribute '
                               f'{self.alias_to} to alias.')
        return ArgAlias(self.alias_to, deprecated=self.deprecated)

    def __repr__(self) -> str:
        return f'Alias({self.alias_to})'


def alias(name: str, deprecated: bool = True) -> ArgAlias:
    return AliasSpec(name, deprecated=deprecated)
    #return Alias(name)


def is_alias(arg: Union[Tuple[str, Any], Any]) -> bool:
    """Determine if a config argument is an alias of another.

    Args:
        arg (Union[Any,Tuple[str,Any]]): the argument or ('name', argument)

    Returns:
        bool: if `arg` is an alias of another argument.
    """
    # arg is ('name', value)
    if isinstance(arg, tuple):
        return is_alias(arg[1])
    return isinstance(arg, ArgAlias)
