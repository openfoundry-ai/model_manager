from typing import Optional
from enum import StrEnum
from dataclasses import dataclass


class ModelSource(StrEnum):
    HuggingFace = "huggingface"
    Sagemaker = "sagemaker"
    Custom = "custom"


class ModelDestination(StrEnum):
    AWS = "aws"
    # AZURE = "azure"
    # GCP = "gcp"


@dataclass
class ModelConfig():
    model_id: str
    source: ModelSource
    model_version: Optional[str] = None
    location: Optional[str] = None

    def __init__(self, **args):
        # TODO: Validations
        self.__dict__.update(args)


@dataclass
class DeploymentConfig:
    destination: ModelDestination
    instance_type: str
    endpoint_name: Optional[str] = None
    instance_count: Optional[int] = 1
    num_gpus: Optional[int] = None
    quantization: Optional[str] = None

    def __init__(self, **args):
        # TODO: Validations

        self.__dict__.update(args)
