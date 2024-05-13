import uvicorn
import os
from dotenv import dotenv_values
from fastapi import FastAPI
from src.config import get_config_for_endpoint, get_endpoints_for_model
from src.sagemaker.resources import get_sagemaker_endpoint
from src.sagemaker.query_endpoint import make_query_request
from src.schemas.query import Query, ChatCompletion
from src.session import session
from litellm import completion

os.environ["AWS_REGION_NAME"] = session.region_name
app = FastAPI()


class NotDeployedException(Exception):
    pass


@app.get("/endpoint/{endpoint_name}")
def get_endpoint(endpoint_name: str):
    return get_sagemaker_endpoint(endpoint_name)


@app.post("/endpoint/{endpoint_name}/query")
def query_endpoint(endpoint_name: str, query: Query):
    config = get_config_for_endpoint(endpoint_name)
    if query.context is None:
        query.context = ''

    # Support multi-model endpoints
    config = (config.deployment, config.models[0])
    return make_query_request(endpoint_name, query, config)


@app.post("/chat/completions")
def chat_completion(chat_completion: ChatCompletion):
    model_id = chat_completion.model

    # Validate model is for completion tasks
    endpoints = get_endpoints_for_model(model_id)
    if len(endpoints) == 0:
        raise NotDeployedException

    messages = chat_completion.messages
    # Currently using the first available endpoint.
    endpoint_name = endpoints[0].deployment.endpoint_name

    res = completion(
        model=f"sagemaker/{endpoint_name}",
        messages=messages,
        temperature=0.9,
        hf_model_name=model_id,
    )

    return res


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
