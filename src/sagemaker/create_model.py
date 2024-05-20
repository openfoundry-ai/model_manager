import json
from dotenv import dotenv_values
from rich.table import Table
from sagemaker import image_uris, model_uris, script_uris
from sagemaker.huggingface import get_huggingface_llm_image_uri
from sagemaker.huggingface.model import HuggingFaceModel
from sagemaker.jumpstart.model import JumpStartModel
from sagemaker.jumpstart.estimator import JumpStartEstimator
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from sagemaker.s3 import S3Uploader
from src.config import write_config
from src.schemas.model import Model, ModelSource
from src.schemas.deployment import Deployment
from src.session import get_sagemaker_session, get_boto_session
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
    region_name = get_boto_session().region_name
    task = get_hf_task(model)
    model.task = task
    env = {
        'HF_MODEL_ID': model.id,
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
        image_uri=image_uri,
        sagemaker_session=get_sagemaker_session(),
    )

    endpoint_name = get_unique_endpoint_name(
        model.id, deployment.endpoint_name)

    deployment.endpoint_name = endpoint_name

    console.log(
        "Deploying model to AWS. [magenta]This may take up to 10 minutes for very large models.[/magenta] See full logs here:")
    console.print(
        f"https://{region_name}.console.aws.amazon.com/cloudwatch/home#logsV2:log-groups/log-group/$252Faws$252Fsagemaker$252FEndpoints$252F{endpoint_name}")

    with console.status("[bold green]Deploying model...") as status:
        table = Table(show_header=False, header_style="magenta")
        table.add_column("Resource", style="dim")
        table.add_column("Value", style="blue")
        table.add_row("model", model.id)
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
        f"{model.id} is now up and running at the endpoint [blue]{predictor.endpoint_name}")

    write_config(deployment, model)
    return predictor


def deploy_custom_huggingface_model(deployment: Deployment, model: Model):
    region_name = get_boto_session().region_name
    if model.location is None:
        print_error("Missing model source location.")
        return

    s3_path = model.location
    if not is_s3_uri(model.location):
        # Local file. Upload to s3 before deploying
        bucket = get_boto_session().default_bucket()
        s3_path = construct_s3_uri(bucket, f"models/{model.id}")
        with console.status(f"[bold green]Uploading custom {model.id} model to S3 at {s3_path}...") as status:
            try:
                s3_path = S3Uploader.upload(
                    model.location, s3_path)
            except Exception:
                print_error("[red] Model failed to upload to S3")

    endpoint_name = get_unique_endpoint_name(
        model.id, deployment.endpoint_name)

    deployment.endpoint_name = endpoint_name
    model.task = get_model_and_task(model.id)['task']

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
        sagemaker_session=get_sagemaker_session()
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
        f"Custom {model.id} is now up and running at the endpoint [blue]{predictor.endpoint_name}")

    write_config(deployment, model)
    return predictor


def create_and_deploy_jumpstart_model(deployment: Deployment, model: Model):
    region_name = get_boto_session().region_name
    endpoint_name = get_unique_endpoint_name(
        model.id, deployment.endpoint_name)
    deployment.endpoint_name = endpoint_name
    model.task = get_model_and_task(model.id)['task']

    console.log(
        "Deploying model to AWS. [magenta]This may take up to 10 minutes for very large models.[/magenta] See full logs here:")

    console.print(
        f"https://{region_name}.console.aws.amazon.com/cloudwatch/home#logsV2:log-groups/log-group/$252Faws$252Fsagemaker$252FEndpoints$252F{endpoint_name}")

    with console.status("[bold green]Deploying model...") as status:
        table = Table(show_header=False, header_style="magenta")
        table.add_column("Resource", style="dim")
        table.add_column("Value", style="blue")
        table.add_row("model", model.id)
        table.add_row("EC2 instance type", deployment.instance_type)
        table.add_row("Number of instances", str(
            deployment.instance_count))
        console.print(table)

        jumpstart_model = JumpStartModel(
            model_id=model.id,
            instance_type=deployment.instance_type,
            role=SAGEMAKER_ROLE,
            sagemaker_session=get_sagemaker_session()
        )

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
        f"{model.id} is now up and running at the endpoint [blue]{predictor.endpoint_name}")

    return predictor
