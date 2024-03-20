from src.sagemaker_helpers.search_sagemaker_jumpstart_models import Frameworks
from huggingface_hub import HfApi
from .rich_utils import print_error


def is_sagemaker_model(endpoint_name: str) -> bool:
    # quick hack
    return endpoint_name.find("--") == -1


def get_sagemaker_framework_and_task(endpoint_or_model_name: str):
    if not is_sagemaker_model(endpoint_or_model_name):
        return None
    components = endpoint_or_model_name.split('-')
    framework, task = components[:2]
    return (framework, task)


def get_hugging_face_pipeline_task(model_name: str):
    hf_api = HfApi()
    try:
        model_info = hf_api.model_info(model_name)
        task = model_info.pipeline_tag
    except Exception:
        print_error("Model not found, please try another.")
        return None

    return task


def get_model_name_from_hugging_face_endpoint(endpoint_name: str):
    endpoint_name = endpoint_name.replace("--", "/")
    split = endpoint_name.split('-')
    return '-'.join(split[:-1])
