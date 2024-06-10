from typing import Union

class ObjectLikeDictionary(dict):
    def __init__(self, **kwargs):
        super().__init__(self, **kwargs)

    def __getattr__(self, name):
        return self[name]


def add_attr_interface(element: Union[dict, list, tuple]):

    if isinstance(element, list):
        return [add_attr_interface(e) for e in element]
    elif isinstance(element, tuple):
        return tuple(add_attr_interface(e) for e in element)
    elif isinstance(element, dict):
        return ObjectLikeDictionary(**{k: add_attr_interface(v) for k,v in element.items()})
    elif isinstance(element, (int, str, float, type(None))):
        return element
    else:
        raise Exception(f"type {type(element)} is not supported")
