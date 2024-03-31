import json
from dotenv import dotenv_values
from rich.table import Table
from sagemaker import image_uris, model_uris, script_uris
from sagemaker.huggingface import get_huggingface_llm_image_uri
from sagemaker.huggingface.model import HuggingFaceModel
from sagemaker.jumpstart.model import JumpStartModel
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from sagemaker.s3 import S3Uploader
from src.config import write_config
from src.schemas.model import Model, ModelSource
from src.schemas.deployment import Deployment, Destination
from src.session import session, sagemaker_session
from src.console import console
from src.utils.aws_utils import construct_s3_uri, is_s3_uri
from src.utils.rich_utils import print_error, print_success
from src.utils.model_utils import get_unique_endpoint_name, get_model_and_task
from src.huggingface import HuggingFaceTask
from src.huggingface.hf_hub_api import get_hf_task

HUGGING_FACE_HUB_TOKEN = dotenv_values(".env").get("HUGGING_FACE_HUB_KEY")
SAGEMAKER_ROLE = dotenv_values(".env")["SAGEMAKER_ROLE"]


# TODO: Consolidate
def deploy_model(deployment: Deployment, model: Model):
    match model.source:
        case ModelSource.HuggingFace:
            deploy_huggingface_model(deployment, model)
        case ModelSource.Sagemaker:
            create_and_deploy_jumpstart_model(deployment, model)
        case ModelSource.Custom:
            deploy_custom_huggingface_model(deployment, model)


def deploy_huggingface_model(deployment: Deployment, model: Model):
    region_name = session.region_name
    task = get_hf_task(model)
    model.task = task
    env = {
        'HF_MODEL_ID': model.model_id,
        'HF_TASK': task,
    }

    if HUGGING_FACE_HUB_TOKEN is not None:
        env['HUGGING_FACE_HUB_TOKEN'] = HUGGING_FACE_HUB_TOKEN

    image_uri = None
    if deployment.num_gpus:
        env['SM_NUM_GPUS'] = json.dumps(deployment.num_gpus)

    if deployment.quantization:
        env['HF_MODEL_QUANTIZE'] = deployment.quantization

    if task == HuggingFaceTask.TextGeneration:
        # use TGI imageq if llm.
        image_uri = get_huggingface_llm_image_uri(
            "huggingface",
            version="1.4.2"
        )

    huggingface_model = HuggingFaceModel(
        env=env,
        role=SAGEMAKER_ROLE,
        transformers_version="4.37",
        pytorch_version="2.1",
        py_version="py310",
        image_uri=image_uri
    )

    endpoint_name = get_unique_endpoint_name(
        model.model_id, deployment.endpoint_name)

    deployment.endpoint_name = endpoint_name
    deployment.destination = Destination.AWS.value

    console.log(
        "Deploying model to AWS. [magenta]This may take up to 10 minutes for very large models.[/magenta] See full logs here:")
    console.print(
        f"https://{region_name}.console.aws.amazon.com/cloudwatch/home#logsV2:log-groups/log-group/$252Faws$252Fsagemaker$252FEndpoints$252F{endpoint_name}")

    with console.status("[bold green]Deploying model...") as status:
        table = Table(show_header=False, header_style="magenta")
        table.add_column("Resource", style="dim")
        table.add_column("Value", style="blue")
        table.add_row("model", model.model_id)
        table.add_row("EC2 instance type", deployment.instance_type)
        table.add_row("Number of instances", str(
            deployment.instance_count))
        table.add_row("task", task)
        console.print(table)

        try:
            predictor = huggingface_model.deploy(
                initial_instance_count=deployment.instance_count,
                instance_type=deployment.instance_type,
                endpoint_name=endpoint_name,
            )
        except Exception:
            console.print_exception()
            quit()

    print_success(
        f"{model.model_id} is now up and running at the endpoint [blue]{predictor.endpoint_name}")

    write_config(deployment, model)
    return predictor


def deploy_custom_huggingface_model(deployment: Deployment, model: Model):
    region_name = session.region_name
    if model.location is None:
        print_error("Missing model source location.")
        return

    s3_path = model.location
    if not is_s3_uri(model.location):
        # Local file. Upload to s3 before deploying
        bucket = sagemaker_session.default_bucket()
        s3_path = construct_s3_uri(bucket, f"models/{model.model_id}")
        with console.status(f"[bold green]Uploading custom {model.model_id} model to S3 at {s3_path}...") as status:
            try:
                s3_path = S3Uploader.upload(
                    model.location, s3_path)
            except Exception:
                print_error("[red] Model failed to upload to S3")

    endpoint_name = get_unique_endpoint_name(
        model.model_id, deployment.endpoint_name)

    deployment.endpoint_name = endpoint_name
    deployment.destination = Destination.AWS.value

    console.log(
        "Deploying model to AWS. [magenta]This may take up to 10 minutes for very large models.[/magenta] See full logs here:")
    console.print(
        f"https://{region_name}.console.aws.amazon.com/cloudwatch/home#logsV2:log-groups/log-group/$252Faws$252Fsagemaker$252FEndpoints$252F{endpoint_name}")

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
        table.add_row("EC2 instance type", deployment.instance_type)
        table.add_row("Number of instances", str(
            deployment.instance_count))
        console.print(table)

        try:
            predictor = huggingface_model.deploy(
                initial_instance_count=deployment.instance_count,
                instance_type=deployment.instance_type,
                endpoint_name=endpoint_name
            )
        except Exception:
            console.print_exception()
            quit()

    print_success(
        f"Custom {model.model_id} is now up and running at the endpoint [blue]{predictor.endpoint_name}")

    write_config(deployment, model)
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


def create_and_deploy_jumpstart_model(deployment: Deployment, model: Model):
    region_name = session.region_name
    endpoint_name = get_unique_endpoint_name(
        model.model_id, deployment.endpoint_name)
    deployment.endpoint_name = endpoint_name
    deployment.destination = Destination.AWS.value
    model.task = get_model_and_task(model.model_id)['task']

    console.log(
        "Deploying model to AWS. [magenta]This may take up to 10 minutes for very large models.[/magenta] See full logs here:")

    console.print(
        f"https://{region_name}.console.aws.amazon.com/cloudwatch/home#logsV2:log-groups/log-group/$252Faws$252Fsagemaker$252FEndpoints$252F{endpoint_name}")

    with console.status("[bold green]Deploying model...") as status:
        table = Table(show_header=False, header_style="magenta")
        table.add_column("Resource", style="dim")
        table.add_column("Value", style="blue")
        table.add_row("model", model.model_id)
        table.add_row("EC2 instance type", deployment.instance_type)
        table.add_row("Number of instances", str(
            deployment.instance_count))
        console.print(table)

        jumpstart_model = JumpStartModel(
            model_id=model.model_id, instance_type=deployment.instance_type, role=SAGEMAKER_ROLE)

        # Attempt to deploy to AWS
        try:
            predictor = jumpstart_model.deploy(
                initial_instance_count=deployment.instance_count,
                instance_type=deployment.instance_type,
                endpoint_name=endpoint_name,
                accept_eula=True
            )
            pass
        except Exception:
            console.print_exception()
            quit()

    write_config(deployment, model)
    print_success(
        f"{model.model_id} is now up and running at the endpoint [blue]{predictor.endpoint_name}")

    return predictor
