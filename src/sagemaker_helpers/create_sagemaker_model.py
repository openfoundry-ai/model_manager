import datetime
import json
from dotenv import dotenv_values
from huggingface_hub import HfApi
from huggingface_hub import login
from rich.table import Table
from sagemaker import image_uris, model_uris, script_uris
from sagemaker.huggingface import get_huggingface_llm_image_uri
from sagemaker.huggingface.model import HuggingFaceModel
from sagemaker.jumpstart.model import JumpStartModel
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from src.session import session, sagemaker_session
from src.console import console
from src.utils.rich_utils import print_error, print_success
from . import HuggingFaceTask, EC2Instance

SAGEMAKER_ROLE = dotenv_values(".env")["SAGEMAKER_ROLE"]
HUGGING_FACE_HUB_TOKEN = dotenv_values(".env").get("HUGGING_FACE_HUB_KEY")


def deploy_huggingface_model(model_id, instance_type: str, instance_count: int = 1):
    hf_api = HfApi()
    region_name = session.region_name
    try:
        model_info = hf_api.model_info(model_id, token=HUGGING_FACE_HUB_TOKEN)
        task = model_info.pipeline_tag
        if model_info.transformers_info is not None and model_info.transformers_info.pipeline_tag is not None:
            task = model_info.transformers_info.pipeline_tag
    except Exception:
        console.print_exception()
        print_error("Model not found, please try another.")
        return

    env = {
        'HF_MODEL_ID': model_id,
        'HF_TASK': task,
    }

    if HUGGING_FACE_HUB_TOKEN is not None:
        env['HUGGING_FACE_HUB_TOKEN'] = HUGGING_FACE_HUB_TOKEN

    image_uri = None
    if task == HuggingFaceTask.TextGeneration and instance_type == EC2Instance.LARGE:
        env['SM_NUM_GPUS'] = json.dumps(4)
        env['HF_MODEL_QUANTIZE'] = "bitsandbytes"
        image_uri = get_huggingface_llm_image_uri(
            "huggingface",
            version="1.0.3"
        )

    huggingface_model = HuggingFaceModel(
        env=env,
        role=SAGEMAKER_ROLE,
        transformers_version="4.37",
        pytorch_version="2.1",
        py_version="py310",
        image_uri=image_uri
    )

    dt_string = datetime.datetime.now().strftime("%Y%m%d%H%M")
    # Endpoint name must be < 63 characters
    model_string = model_id.replace(
        "/", "--").replace("_", "-").replace(".", "")[:50]
    endpoint_name = f"{model_string}-{dt_string}"

    console.log(
        "Deploying model to AWS. [magenta]This may take up to 10 minutes for very large models.[/magenta] See full logs here:")
    console.print(
        f"https://{region_name}.console.aws.amazon.com/cloudwatch/home#logsV2:log-groups/log-group/$252Faws$252Fsagemaker$252FEndpoints$252F{endpoint_name}")

    with console.status("[bold green]Deploying model...") as status:
        table = Table(show_header=False, header_style="magenta")
        table.add_column("Resource", style="dim")
        table.add_column("Value", style="blue")
        table.add_row("model", model_id)
        table.add_row("EC2 instance type", instance_type)
        table.add_row("Number of instances", str(instance_count))
        table.add_row("task", task)
        console.print(table)

        try:
            predictor = huggingface_model.deploy(
                initial_instance_count=instance_count,
                instance_type=instance_type,
                endpoint_name=endpoint_name,
            )
        except Exception:
            console.print_exception()
            quit()

    print_success(
        f"{model_id} is now up and running at the endpoint [blue]{predictor.endpoint_name}")
    return predictor


def deploy_custom_huggingface_model(s3_path, base_model, instance_type: str, instance_count: int = 1):
    region_name = session.region_name
    dt_string = datetime.datetime.now().strftime("%Y%m%d%H%M")
    # Endpoint name must be < 63 characters
    model_string = base_model.replace(
        "/", "--").replace("_", "-").replace(".", "")[:43]
    endpoint_name = f"custom-{model_string}-{dt_string}"

    # create Hugging Face Model Class
    huggingface_model = HuggingFaceModel(
        # path to your trained sagemaker model
        model_data=s3_path,
        role=SAGEMAKER_ROLE,  # iam role with permissions to create an Endpoint
        transformers_version="4.37",
        pytorch_version="2.1",
        py_version="py310",
    )

    with console.status("[bold green]Deploying model...") as status:
        table = Table(show_header=False, header_style="magenta")
        table.add_column("Resource", style="dim")
        table.add_column("Value", style="blue")
        table.add_row("S3 Path", s3_path)
        table.add_row("EC2 instance type", instance_type)
        table.add_row("Number of instances", str(instance_count))
        console.print(table)

        try:
            predictor = huggingface_model.deploy(
                initial_instance_count=instance_count,
                instance_type=instance_type,
                endpoint_name=endpoint_name
            )
        except Exception:
            console.print_exception()
            quit()

    print_success(
        f"Custom {base_model} is now up and running at the endpoint [blue]{predictor.endpoint_name}")
    return predictor


def create_sagemaker_model(model_id: str, model_version: str, instance_type, instance_count, is_train: bool = False):
    scope = "training" if is_train else "inference"
    entry_point = "train.py" if is_train else "inference.py"

    # Retrieve the URIs of the JumpStart resources
    console.log("fetching model metadata")
    base_model_uri = model_uris.retrieve(
        model_id=model_id, model_version=model_version, model_scope=scope, sagemaker_session=sagemaker_session
    )
    script_uri = script_uris.retrieve(
        model_id=model_id, model_version=model_version, script_scope=scope, sagemaker_session=sagemaker_session
    )
    image_uri = image_uris.retrieve(
        region=None,
        framework=None,
        image_scope=scope,
        model_id=model_id,
        model_version=model_version,
        instance_type=instance_type,
        sagemaker_session=sagemaker_session
    )

    table = Table(show_header=False, header_style="magenta")
    table.add_column("Resource", style="dim")
    table.add_column("Value", style="blue")
    table.add_row("model", model_id)
    table.add_row("model_version", model_version)
    table.add_row("EC2 instance type", instance_type)
    table.add_row("Number of instances", str(instance_count))
    table.add_row("base_model_uri", base_model_uri)
    table.add_row("script_uri", script_uri)
    table.add_row("image_uri", image_uri)
    console.print(table)

    # Create the SageMaker model instance
    model = JumpStartModel(
        image_uri=image_uri,
        model_data=base_model_uri,
        source_dir=script_uri,
        entry_point=entry_point,
        role=SAGEMAKER_ROLE,
        predictor_cls=Predictor,
        enable_network_isolation=True,
        sagemaker_session=sagemaker_session
    )

    return model


def create_and_deploy_jumpstart_model(model_id: str, model_version: str, instance_type: str, instance_count: int = 1):
    region_name = session.region_name
    dt_string = datetime.datetime.now().strftime("%Y%m%d%H%M")
    # Endpoint name must be < 63 characters
    model_string = model_id[:50]
    endpoint_name = f"{model_string}-{dt_string}"

    console.log(
        "Deploying model to AWS. [magenta]This may take up to 10 minutes for very large models.[/magenta] See full logs here:")

    console.print(
        f"https://{region_name}.console.aws.amazon.com/cloudwatch/home#logsV2:log-groups/log-group/$252Faws$252Fsagemaker$252FEndpoints$252F{endpoint_name}")

    with console.status("[bold green]Deploying model...") as status:
        model = JumpStartModel(
            model_id=model_id, instance_type=instance_type, role=SAGEMAKER_ROLE)

        # Attempt to deploy to AWS
        try:
            predictor = model.deploy(
                initial_instance_count=instance_count,
                instance_type=instance_type,
                endpoint_name=endpoint_name,
                accept_eula=True
            )
        except Exception:
            console.print_exception()
            quit()

    print_success(
        f"{model_id} is now up and running at the endpoint [blue]{predictor.endpoint_name}")
    return predictor
