from typing import *

AutogradFunction = Type

class PrefixBundle:
    def __init__(self, bundle: Dict = {  }, default = None) -> None:
        self.__dict__["_bundle"] = bundle
        self.__dict__["_default"] = default

    def __setattr__(self, name: str, value: Any) -> None:
        self._bundle[name] = value

    def __getattr__(self, name: str) -> Any:
        while len(name) > 0:
            if name in self._bundle:
                return self._bundle[name]
            else:
                name = name[0:-1]
        
        if self._default is not None:
            return self._default
        else:
            raise KeyError("item not found")
