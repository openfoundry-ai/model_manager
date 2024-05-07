import yaml
from src.yaml import loader, dumper
from typing import Optional
from enum import StrEnum
from pydantic import BaseModel


class Destination(StrEnum):
    AWS = "aws"
    # AZURE = "azure"
    # GCP = "gcp"


class Deployment(BaseModel):
    destination: Destination
    instance_type: str
    endpoint_name: Optional[str] = None
    instance_count: Optional[int] = 1
    num_gpus: Optional[int] = None
    quantization: Optional[str] = None


def deployment_representer(dumper: yaml.SafeDumper, deployment: Deployment) -> yaml.nodes.MappingNode:
    return dumper.represent_mapping("!Deployment", {
        "destination": deployment.destination.value,
        "instance_type": deployment.instance_type,
        "endpoint_name": deployment.endpoint_name,
        "instance_count": deployment.instance_count,
        "num_gpus": deployment.num_gpus,
        "quantization": deployment.quantization,
    })


def deployment_constructor(loader: yaml.SafeLoader, node: yaml.nodes.ScalarNode) -> Deployment:
    return Deployment(**loader.construct_mapping(node))


dumper.add_representer(Deployment, deployment_representer)
loader.add_constructor(u'!Deployment', deployment_constructor)
