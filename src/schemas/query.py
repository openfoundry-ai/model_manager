
from pydantic import BaseModel
from typing import Optional


class QueryParameters(BaseModel):
    max_length: Optional[int] = None
    max_new_tokens: Optional[int] = None
    repetition_penalty: Optional[float] = None
    temperature: Optional[float] = None
    top_k: Optional[float] = None
    top_p: Optional[float] = None


class Query(BaseModel):
    query: str
    context: Optional[str] = None
    parameters: Optional[QueryParameters] = None
