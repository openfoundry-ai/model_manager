from pydantic import BaseModel, SecretStr
from enum import StrEnum


class SecretKeys(StrEnum):
    AWSAccessKey = 'AWS_ACCESS_KEY'
    AWSSecretAccessKey = 'AWS_SECRET_ACCESS_KEY'
    HuggingFaceHubKey = 'HF_HUB_KEY'


class Secret(BaseModel):
    key: SecretKeys
    value: SecretStr


class Secrets(BaseModel):
    secrets: list[Secret]
