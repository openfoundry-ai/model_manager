import yaml
from src.yaml import loader, dumper
from typing import Optional, Union, Dict
from enum import StrEnum
from dataclasses import dataclass
from src.huggingface import HuggingFaceTask
from src.sagemaker_helpers import SagemakerTask


class ModelSource(StrEnum):
    HuggingFace = "huggingface"
    Sagemaker = "sagemaker"
    Custom = "custom"


@dataclass
class Model():
    model_id: str
    source: ModelSource
    task: Optional[Union[HuggingFaceTask, SagemakerTask]] = None
    model_version: Optional[str] = None
    location: Optional[str] = None
    predict: Optional[Dict[str, str]] = None

    def __init__(self, **args):
        # TODO: Validations
        self.__dict__.update(args)


def model_representer(dumper: yaml.SafeDumper, model: Model) -> yaml.nodes.MappingNode:
    return dumper.represent_mapping("!Model", {
        "model_id": model.model_id,
        "source": model.source,
        "task": model.task,
        "model_version": model.model_version,
        "location": model.location,
        "predict": model.predict,
    })


def model_constructor(loader: yaml.SafeLoader, node: yaml.nodes.MappingNode) -> Model:
    """Construct an employee."""
    return Model(**loader.construct_mapping(node))


dumper.add_representer(Model, model_representer)
loader.add_constructor(u'!Model', model_constructor)
