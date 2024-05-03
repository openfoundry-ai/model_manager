import logging
import os
import sagemaker
from botocore.exceptions import ClientError
from datasets import load_dataset
from rich import print
from rich.table import Table
from sagemaker.jumpstart.estimator import JumpStartEstimator
from src.console import console
from src.schemas.model import Model, ModelSource
from src.schemas.training import Training
from src.session import sagemaker_session
from src.utils.aws_utils import is_s3_uri
from src.utils.rich_utils import print_success, print_error
from transformers import AutoTokenizer

from dotenv import load_dotenv
load_dotenv()
SAGEMAKER_ROLE = os.environ.get("SAGEMAKER_ROLE")


def prep_hf_data(s3_bucket: str, dataset_name_or_path: str, model: Model):
    train_dataset, test_dataset = load_dataset(
        dataset_name_or_path, split=["train", "test"])
    tokenizer = AutoTokenizer.from_pretrained(model.id)

    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True)

    # tokenize train and test datasets
    train_dataset = train_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)

    # set dataset format for PyTorch
    train_dataset = train_dataset.rename_column("label", "labels")
    train_dataset.set_format(
        "torch", columns=["input_ids", "attention_mask", "labels"])
    test_dataset = test_dataset.rename_column("label", "labels")
    test_dataset.set_format(
        "torch", columns=["input_ids", "attention_mask", "labels"])

    # save train_dataset to s3
    training_input_path = f's3://{s3_bucket}/datasets/train'
    train_dataset.save_to_disk(training_input_path)

    # save test_dataset to s3
    test_input_path = f's3://{s3_bucket}/datasets/test'
    test_dataset.save_to_disk(test_input_path)

    return training_input_path, test_input_path


def train_model(training: Training, model: Model, estimator):
    # TODO: Accept hf datasets or local paths to upload to s3
    if not is_s3_uri(training.training_input_path):
        raise Exception("Training data needs to be uploaded to s3")

    # TODO: Implement training, validation, and test split or accept a directory of files
    training_dataset_s3_path = training.training_input_path

    table = Table(show_header=False, header_style="magenta")
    table.add_column("Resource", style="dim")
    table.add_column("Value", style="blue")
    table.add_row("model", model.id)
    table.add_row("model_version", model.version)
    table.add_row("base_model_uri", estimator.model_uri)
    table.add_row("image_uri", estimator.image_uri)
    table.add_row("EC2 instance type", training.instance_type)
    table.add_row("Number of instances", str(training.instance_count))
    console.print(table)

    estimator.fit({"training": training_dataset_s3_path})

    predictor = estimator.deploy(
        initial_instance_count=training.instance_count, instance_type=training.instance_type)

    print_success(
        f"Trained model {model.id} is now up and running at the endpoint [blue]{predictor.endpoint_name}")


def fine_tune_model(training: Training, model: Model):
    estimator = None
    match model.source:
        case ModelSource.Sagemaker:
            hyperparameters = get_hyperparameters_for_model(training, model)
            estimator = JumpStartEstimator(
                model_id=model.id,
                model_version=model.version,
                instance_type=training.instance_type,
                instance_count=training.instance_count,
                output_path=training.output_path,
                environment={"accept_eula": "true"},
                role=SAGEMAKER_ROLE,
                sagemaker_session=sagemaker_session,
                hyperparameters=hyperparameters
            )
        case ModelSource.HuggingFace:
            raise NotImplementedError
        case ModelSource.Custom:
            raise NotImplementedError

    try:
        print_success("Enqueuing training job")
        res = train_model(training, model, estimator)
    except ClientError as e:
        logging.error(e)
        print_error("Training job enqueue fail")
        return False


def get_hyperparameters_for_model(training: Training, model: Model):
    hyperparameters = sagemaker.hyperparameters.retrieve_default(
        model_id=model.id, model_version=model.version)

    if training.hyperparameters is not None:
        hyperparameters.update(
            (k, v) for k, v in training.hyperparameters.model_dump().items() if v is not None)
    return hyperparameters
