from pathlib import Path

from yaml import load

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

CONFIG_DIR = Path("config")


class Config:
    def __init__(self, name: str, config_file_name: str = "config.yaml"):
        self.name = name
        self._config = load((CONFIG_DIR / config_file_name), Loader=Loader)

    def get(self, key: str):
        return self._config.get(key)
