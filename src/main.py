import inquirer
import logging
import threading
from InquirerPy import prompt
from src.sagemaker import EC2Instance
from src.sagemaker.create_model import deploy_model
from src.sagemaker.delete_model import delete_sagemaker_model
from src.sagemaker.resources import list_sagemaker_endpoints, select_instance, list_service_quotas_async
from src.sagemaker.query_endpoint import make_query_request
from src.sagemaker.search_jumpstart_models import search_sagemaker_jumpstart_model
from src.utils.rich_utils import print_error, print_success
from src.schemas.deployment import Deployment, Destination
from src.schemas.model import Model, ModelSource
from src.schemas.query import Query
from src.config import get_config_for_endpoint
from enum import StrEnum
from rich import print


class Actions(StrEnum):
    LIST = "Show active model endpoints"
    DEPLOY = "Deploy a model endpoint"
    DELETE = "Delete a model endpoint"
    QUERY = "Query a model endpoint"
    EXIT = "Quit"
    # TRAIN = "fine tune a model"


def main(args, loglevel):
    logging.basicConfig(format="%(levelname)s: %(message)s", level=loglevel)

    print("[magenta]Model Manager by OpenFoundry.")
    print("[magenta]Star us on Github ☆! [blue]https://github.com/openfoundry-ai/model_manager")

    # list_service_quotas is a pretty slow API and it's paginated.
    # Use async here and store the result in instances
    instances = []
    instance_thread = threading.Thread(
        target=list_service_quotas_async, args=[instances])
    instance_thread.start()

    while True:
        active_endpoints = list_sagemaker_endpoints(onlyActive=True)
        questions = [
            inquirer.List(
                'action',
                message="What would you like to do?",
                choices=[e.value for e in Actions]
            )
        ]
        answers = inquirer.prompt(questions)
        if (answers is None):
            break

        action = answers['action']

        match action:
            case Actions.LIST:
                if len(active_endpoints) != 0:
                    [print(f"[blue]{endpoint['EndpointName']}[/blue] running on an [green]{endpoint['InstanceType']}[/green] instance")
                     for endpoint in active_endpoints]
                    print('\n')
                else:
                    print_error('No active endpoints.\n')
            case Actions.DEPLOY:
                build_and_deploy_model(instances, instance_thread)
            case Actions.DELETE:
                if (len(active_endpoints) == 0):
                    print_success("No Endpoints to delete!")
                    continue

                questions = [
                    inquirer.Checkbox('endpoints',
                                      message="Which endpoints would you like to delete? (space to select)",
                                      choices=[endpoint['EndpointName']
                                               for endpoint in active_endpoints]
                                      )
                ]
                answers = inquirer.prompt(questions)
                if (answers is None):
                    continue

                endpoints_to_delete = answers['endpoints']
                delete_sagemaker_model(endpoints_to_delete)
            case Actions.QUERY:
                if (len(active_endpoints) == 0):
                    print_success("No Endpoints to query!")
                    continue

                questions = [
                    {
                        "type": "list",
                        "message": "Which endpoint would you like to query?",
                        "name": 'endpoint',
                        "choices": [endpoint['EndpointName'] for endpoint in active_endpoints]
                    },
                    {
                        "type": "input",
                        "message": "What would you like to query?",
                        "name": 'query'
                    }
                ]
                answers = prompt(questions)
                if (answers is None):
                    continue

                endpoint = answers['endpoint']
                query = Query(query=answers['query'])
                config = get_config_for_endpoint(endpoint)

                # support multi-model endpoints
                config = (config.deployment, config.models[0])
                make_query_request(endpoint, query, config)
            case Actions.EXIT:
                quit()

    print_success("Goodbye!")


class ModelType(StrEnum):
    SAGEMAKER = "Deploy a Sagemaker model"
    HUGGINGFACE = "Deploy a Hugging Face model"
    CUSTOM = "Deploy a custom model"


def build_and_deploy_model(instances, instance_thread):
    questions = [
        inquirer.List(
            'model_type',
            message="Choose a model type:",
            choices=[model_type.value for model_type in ModelType]
        )
    ]
    answers = inquirer.prompt(questions)
    if answers is None:
        return

    model_type = answers['model_type']

    match model_type:
        case ModelType.SAGEMAKER:
            models = []
            while len(models) == 0:
                models = search_sagemaker_jumpstart_model()
                if models is None:
                    quit()

            questions = [
                {
                    "type": "fuzzy",
                    "name": "model_id",
                    "message": "Choose a model. Search by task (e.g. eqa) or model name (e.g. llama):",
                    "choices": models,
                    "match_exact": True,
                }
            ]
            answers = prompt(questions)
            if (answers is None):
                return

            model_id, model_version = answers['model_id'], "2.*"
            instance_thread.join()
            instance_type = select_instance(instances)

            model = Model(
                id=model_id,
                model_version=model_version,
                source=ModelSource.Sagemaker
            )

            deployment = Deployment(
                instance_type=instance_type,
                destination=Destination.AWS,
                num_gpus=4 if instance_type == EC2Instance.LARGE else 1
            )

            predictor = deploy_model(deployment=deployment, model=model)
        case ModelType.HUGGINGFACE:
            questions = [
                inquirer.Text(
                    'model_id',
                    message="Enter the exact model name from huggingface.co (e.g. google-bert/bert-base-uncased)",
                )
            ]
            answers = inquirer.prompt(questions)
            if answers is None:
                return

            model_id = answers['model_id']
            instance_thread.join()
            instance_type = select_instance(instances)

            model = Model(
                id=model_id,
                source=ModelSource.HuggingFace
            )

            deployment = Deployment(
                instance_type=instance_type,
                destination=Destination.AWS,
            )

            predictor = deploy_model(deployment=deployment, model=model)

        case ModelType.CUSTOM:
            questions = [
                {
                    "type": "input",
                    "message": "What is the local path or S3 URI of the model?",
                    "name": "path"
                },
                {
                    "type": "input",
                    "message": "What is the base model that was fine tuned? (e.g. google-bert/bert-base-uncased)",
                    "name": "base_model"
                }
            ]
            answers = prompt(questions)
            local_path = answers['path']
            base_model = answers['base_model']

            instance_thread.join()
            instance_type = select_instance(instances)

            model = Model(
                id=base_model,
                source=ModelSource.Custom,
                location=local_path
            )

            deployment = Deployment(
                instance_type=instance_type,
                destination=Destination.AWS
            )
            deploy_model(deployment=deployment, model=model)
