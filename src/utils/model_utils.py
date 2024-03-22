from src.sagemaker_helpers.search_sagemaker_jumpstart_models import Frameworks
from huggingface_hub import HfApi
from .rich_utils import print_error
from difflib import SequenceMatcher
from dotenv import dotenv_values
HUGGING_FACE_HUB_TOKEN = dotenv_values(".env").get("HUGGING_FACE_HUB_KEY")


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
        model_info = hf_api.model_info(
            model_name, token=HUGGING_FACE_HUB_TOKEN)
        task = model_info.pipeline_tag
    except Exception:
        print_error("Model not found, please try another.")
        return None

    return task


def get_model_name_from_hugging_face_endpoint(endpoint_name: str):
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
