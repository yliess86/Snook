import re
import yaml

from typing import Any
from typing import Dict


class AttributeDict(dict):
    def __init__(self, mapping: Dict = None):
        super(AttributeDict, self).__init__()
        if mapping is not None:
            for key, value in mapping.items():
                self.__setitem__(key, value)

    def __setitem__(self, key: str, value: Any):
        if isinstance(value, dict):
            value = AttributeDict(value)
        elif isinstance(value, list):
            value = [AttributeDict(v) if isinstance(v, dict) else v for v in value]
        elif isinstance(value, str) and "e" in value:
            if value in re.findall(r"-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?", value):
                value = float(value)
        super(AttributeDict, self).__setitem__(key, value)
        self.__dict__[key] = value

    def __getattr__(self, item: Any):
        try:
            return self.__getitem__(item)
        except KeyError:
            raise AttributeError(item)

    __setattr__ = __setitem__


class Config(AttributeDict):
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, "r") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        config = cls(data)
        return config