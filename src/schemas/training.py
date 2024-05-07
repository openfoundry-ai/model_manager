import yaml
from pydantic import BaseModel
from typing import Optional
from src.schemas.deployment import Destination
from src.yaml import loader, dumper


class Hyperparameters(BaseModel):
    epochs: Optional[int] = None
    per_device_train_batch_size: Optional[int] = None
    learning_rate: Optional[float] = None


class Training(BaseModel):
    destination: Destination
    instance_type: str
    instance_count: int
    training_input_path: str
    output_path: Optional[str] = None
    hyperparameters: Optional[Hyperparameters] = None


def training_representer(dumper: yaml.SafeDumper, training: Training) -> yaml.nodes.MappingNode:
    return dumper.represent_mapping("!Training", {
        "destination": training.destination.value,
        "instance_type": training.instance_type,
        "instance_count": training.instance_count,
        "output_path": training.output_path,
        "training_input_path": training.training_input_path,
        "hyperparameters": training.hyperparameters,
    })


def training_constructor(loader: yaml.SafeLoader, node: yaml.nodes.MappingNode) -> Training:
    return Training(**loader.construct_mapping(node))


def hyperparameters_constructor(loader: yaml.SafeLoader, node: yaml.nodes.MappingNode) -> Hyperparameters:
    return Hyperparameters(**loader.construct_mapping(node))


dumper.add_representer(Training, training_representer)
loader.add_constructor(u'!Training', training_constructor)
loader.add_constructor(u'!Hyperparameters', hyperparameters_constructor)
