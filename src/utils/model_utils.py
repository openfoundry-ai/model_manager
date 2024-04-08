import datetime
from difflib import SequenceMatcher
from dotenv import dotenv_values
from huggingface_hub import HfApi
from src.utils.rich_utils import print_error
from src.sagemaker import SagemakerTask
from src.schemas.deployment import Deployment
from src.schemas.model import Model, ModelSource
from src.schemas.query import Query
from src.session import sagemaker_session
from typing import Dict, Tuple, Optional
HUGGING_FACE_HUB_TOKEN = dotenv_values(".env").get("HUGGING_FACE_HUB_KEY")


def get_unique_endpoint_name(model_id: str, endpoint_name: str = None):
    dt_string = datetime.datetime.now().strftime("%Y%m%d%H%M")

    if not endpoint_name:
        # Endpoint name must be < 63 characters
        model_string = model_id.replace(
            "/", "--").replace("_", "-").replace(".", "")[:50]
        return f"{model_string}-{dt_string}"
    else:
        return f"{endpoint_name[:50]}-{dt_string}"


def is_sagemaker_model(endpoint_name: str, config: Optional[Tuple[Deployment, Model]] = None) -> bool:
    if config is not None:
        _, model = config
        task = get_sagemaker_model_and_task(model.id)['task']

        # check task for custom sagemaker models
        return model.source == ModelSource.Sagemaker or task in SagemakerTask.list()

    # fallback
    return endpoint_name.find("--") == -1


def is_custom_model(endpoint_name: str) -> bool:
    return endpoint_name.startswith('custom')


def get_sagemaker_model_and_task(endpoint_or_model_name: str):
    components = endpoint_or_model_name.split('-')
    framework, task = components[:2]
    model_id = '-'.join(components[:-1])
    return {
        'model_id': model_id,
        'task': task,
    }


def get_hugging_face_pipeline_task(model_name: str):
    hf_api = HfApi()
    try:
        model_info = hf_api.model_info(
            model_name, token=HUGGING_FACE_HUB_TOKEN)
        task = model_info.pipeline_tag
    except Exception:
        print_error("Model not found, please try another.")
        return None

    return task


def get_model_and_task(endpoint_or_model_name: str, config: Optional[Tuple[Deployment, Model]] = None) -> Dict[str, str]:
    if config is not None:
        _, model = config
        return {
            'model_id': model.id,
            'task': model.task
        }

    if (is_sagemaker_model(endpoint_or_model_name)):
        return get_sagemaker_model_and_task(endpoint_or_model_name)
    else:
        model_id = get_model_name_from_hugging_face_endpoint(
            endpoint_or_model_name)
        task = get_hugging_face_pipeline_task(model_id)
        return {
            'model_id': model_id,
            'task': task
        }


def get_model_name_from_hugging_face_endpoint(endpoint_name: str):
    endpoint_name = endpoint_name.removeprefix("custom-")
    endpoint_name = endpoint_name.replace("--", "/")
    author, rest = endpoint_name.split("/")

    # remove datetime
    split = rest.split('-')
    fuzzy_model_name = '-'.join(split[:-1])

    # get first token
    search_term = fuzzy_model_name.split('-')[0]

    hf_api = HfApi()
    results = hf_api.list_models(search=search_term, author=author)

    # find results that closest match our fuzzy model name
    results_to_diff = {}
    for result in results:
        results_to_diff[result.id] = SequenceMatcher(
            None, result.id, f"{author}/{fuzzy_model_name}").ratio()

    return max(results_to_diff, key=results_to_diff.get)


def get_text_generation_hyperpameters(config: Optional[Tuple[Deployment, Model]], query: Query = None):
    if query.parameters is not None:
        return query.parameters.model_dump()

    if config is not None and config[1].predict is not None:
        return config[1].predict

    # Defaults
    return {
        "max_new_tokens": 250,
        "top_p": 0.9,
        "temperature": 0.9,
    }
