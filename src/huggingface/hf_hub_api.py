from dotenv import dotenv_values
from src.console import console
from src.huggingface import hf_api
from src.schemas.model import Model
from src.utils.rich_utils import print_error

HUGGING_FACE_HUB_TOKEN = dotenv_values(".env").get("HUGGING_FACE_HUB_KEY")


def get_hf_task(model: Model):
    task = None
    try:
        model_info = hf_api.model_info(
            model.id, token=HUGGING_FACE_HUB_TOKEN)
        task = model_info.pipeline_tag
        if model_info.transformers_info is not None and model_info.transformers_info.pipeline_tag is not None:
            task = model_info.transformers_info.pipeline_tag
    except Exception:
        # better error handling for auth
        console.print_exception()
        print_error("Model not found, please try another.")
        return
    return task
