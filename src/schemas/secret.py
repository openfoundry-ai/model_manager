from pydantic import BaseModel


class Secret(BaseModel):
    key: str
    value: str


class Secrets(BaseModel):
    secrets: list[Secret]
