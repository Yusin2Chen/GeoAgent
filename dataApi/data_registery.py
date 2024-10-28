import enum
import inspect
import typing
from typing import Any, Callable
import datetime
from docstring_parser import parse
from pydantic import BaseModel
from .gee_utils import BBox, matches_interval

def to_json_schema_type(type_name: str) -> str:
    type_map = {
        "str": "string",
        "int": "integer",
        "float": "number",
        "bool": "boolean",
        "None": "null",
        "Any": "any",
        "Dict": "object",
        "List": "array",
        "Optional": "any",
        "datetime": "datetime",
    }
    return type_map.get(type_name, "any")


def parse_annotation(annotation):
    if getattr(annotation, "__origin__", None) == typing.Union:
        types = [t.__name__ if t.__name__ != "NoneType" else "None" for t in annotation.__args__]
        return to_json_schema_type(types[0])
    elif issubclass(annotation, enum.Enum):  # If the annotation is an Enum type
        return "enum", [item.name for item in annotation]  # Return 'enum' and a list of the names of the enum members
    elif getattr(annotation, "__origin__", None) is not None:
        if annotation._name is not None:
            return f"{to_json_schema_type(annotation._name)}[{','.join([to_json_schema_type(i.__name__) for i in annotation.__args__])}]"
        else:
            return f"{to_json_schema_type(annotation.__origin__.__name__)}[{','.join([to_json_schema_type(i.__name__) for i in annotation.__args__])}]"
    else:
        return to_json_schema_type(annotation.__name__)


def get_pydantic_schema(pydantic_obj: BaseModel, visited_models=None) -> dict:
    if visited_models is None:
        visited_models = set()

    if pydantic_obj in visited_models:
        raise ValueError(f"Circular reference detected: {pydantic_obj.__name__}")

    visited_models.add(pydantic_obj)

    schema = pydantic_obj.schema()
    definitions = schema.pop("definitions", {})

    def resolve_schema(schema):
        if "$ref" in schema:
            ref_path = schema["$ref"]
            definition_key = ref_path.split("/")[-1]
            return resolve_schema(definitions[definition_key])
        elif "items" in schema:
            schema["items"] = resolve_schema(schema["items"])
        return schema

    schema = resolve_schema(schema)
    for name, property in schema["properties"].items():
        schema["properties"][name] = resolve_schema(property)

    visited_models.remove(pydantic_obj)

    return schema


class DataRegistry:
    """
    A registry for database that can be called by LLM.
    """

    def __init__(self):
        self.functions = {}

    def to_list(self, query_bbox: BBox, query_interval: tuple[datetime.datetime, datetime.datetime], sensor: str):
        """
        Return a list of (function_name, function) tuples, filtered by bbox and timeinterval if provided.
        """
        if query_bbox and query_interval:
            def _inetset(func):
                bBox_interset = [_bBox.intersects(query_bbox) for _bBox in func['bBox']]
                return any(bBox_interset)

            def _matched(func):
                time_matched = [matches_interval(_time, query_interval) for _time in func['timeInterval']]
                return any(time_matched)

            def _sesnor(func):
                if sensor == None:
                    return True
                return func['sensor'] == sensor

            return [(name, func['func']) for name, func in self.functions.items()
                    if _inetset(func) and _matched(func) and _sesnor(func)]
        return list(self.functions.items())

    def to_list_infos(self, query_bbox: BBox, query_interval: tuple[datetime.datetime, datetime.datetime], sensor: str):
        """
        Return a list of (function_name, function) tuples, filtered by bbox and timeinterval if provided.
        """
        if query_bbox and query_interval:
            def _inetset(func):
                bBox_interset = [_bBox.intersects(query_bbox) for _bBox in func['bBox']]
                return any(bBox_interset)

            def _matched(func):
                time_matched = [matches_interval(_time, query_interval) for _time in func['timeInterval']]
                return any(time_matched)

            def _sesnor(func):
                if sensor == None:
                    return True
                return func['sensor'] == sensor

            return [(name, self.function_schema(func['func'])) for name, func in self.functions.items()
                    if _inetset(func) and _matched(func) and _sesnor(func)]
        return list(self.functions.items())

    def add(self):
        """
        Register a function to the registry, allowing sensor assignment based on the function's local variables after execution.
        """
        def decorator(cls):
            #print(f"Decorating class: {cls.__name__}")  # Confirming which class is being decorated
            instance = cls()

            class Wrapped(cls):
                def __init__(self, registry_instance, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.registry_instance = registry_instance  # Store the reference to DataRegistry
                    # Automatically register all methods starting with "get_"
                    for attr_name in dir(instance):
                        method = getattr(instance, attr_name)
                        if callable(method) and attr_name.startswith("get_"):
                            wrapped_method = self._wrap_method(method, instance)
                            # Registering in DataRegistry instance
                            self.registry_instance.functions[f"{cls.__name__}.{attr_name}"] = wrapped_method

                def _wrap_method(self, method, instance):
                    """
                    Wraps the method to access instance variables (like sensors).
                    """
                    if hasattr(instance, 'bBox'):
                        bBox = instance.bBox  #ToDo
                    else:
                        bBox = None

                    if hasattr(instance, 'timeInterval'):
                        timeInterval = list(instance.timeInterval) #ToDo
                    else:
                        timeInterval = None

                    if hasattr(instance, 'sensor'):
                        sensor = instance.sensor #ToDo
                    else:
                        sensor = None

                    return {'func': method, 'bBox': bBox, 'timeInterval': timeInterval, 'sensor': sensor}

            return lambda *args, **kwargs: Wrapped(self, *args, **kwargs)  # Pass self as the registry instance

        return decorator

    def function_schema(self, func: Callable) -> dict:
        signature = inspect.signature(func)
        docstring = inspect.getdoc(func)
        docstring_parsed = parse(docstring)

        parameters = dict()
        required = []

        for name, param in signature.parameters.items():
            json_type = parse_annotation(param.annotation)

            if isinstance(param.annotation, type) and issubclass(param.annotation, BaseModel):
                param_info = get_pydantic_schema(param.annotation)
                param_info["description"] = param.annotation.__doc__
                required.append(name)
            elif isinstance(json_type, tuple) and json_type[0] == "enum":
                param_info = {
                    "type": "string",
                    "enum": json_type[1],
                    "description": "",
                }
            else:
                param_info = {"type": json_type, "description": ""}

            if json_type != "any" and name != "self" and param.default == inspect.Parameter.empty:
                required.append(name)

            for doc_param in docstring_parsed.params:
                if doc_param.arg_name == name:
                    param_info["description"] = doc_param.description

            parameters[name] = param_info

        function_info = {
            "name": func.__name__,
            "description": docstring_parsed.short_description,
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required,
            },
        }

        return function_info

    def get_all_function_information(self):
        """
        Get all function information from the registry.
        """
        return [self.function_schema(func) for func in self.functions.values()]

    def get(self, name: str) -> Callable[..., Any]:
        """
        Get a function from the registry.
        """
        return self.functions[name]

