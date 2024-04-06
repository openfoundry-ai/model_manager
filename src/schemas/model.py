import yaml
from src.yaml import loader, dumper
from typing import Optional, Union, Dict
from enum import StrEnum
from src.huggingface import HuggingFaceTask
from src.sagemaker_helpers import SagemakerTask
from pydantic import BaseModel


class ModelSource(StrEnum):
    HuggingFace = "huggingface"
    Sagemaker = "sagemaker"
    Custom = "custom"


class Model(BaseModel):
    id: str
    source: ModelSource
    task: Optional[Union[HuggingFaceTask, SagemakerTask]] = None
    version: Optional[str] = None
    location: Optional[str] = None
    predict: Optional[Dict[str, str]] = None


def model_representer(dumper: yaml.SafeDumper, model: Model) -> yaml.nodes.MappingNode:
    return dumper.represent_mapping("!Model", {
        "id": model.id,
        "source": model.source.value,
        "task": model.task,
        "version": model.version,
        "location": model.location,
        "predict": model.predict,
    })


def model_constructor(loader: yaml.SafeLoader, node: yaml.nodes.MappingNode) -> Model:
    """Construct an employee."""
    return Model(**loader.construct_mapping(node))


dumper.add_representer(Model, model_representer)
loader.add_constructor(u'!Model', model_constructor)
