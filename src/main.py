import inquirer
import logging
logging.getLogger("sagemaker.config").setLevel(logging.WARNING)
logging.getLogger("botocore.credentials").setLevel(logging.WARNING)
import sagemaker
import boto3
from .sagemaker_helpers.create_sagemaker_model import create_and_deploy_jumpstart_model, deploy_huggingface_model
from .sagemaker_helpers.delete_sagemaker_model import delete_sagemaker_model
from .sagemaker_helpers.sagemaker_resources import list_sagemaker_endpoints, select_instance
from .sagemaker_helpers.query_sagemaker_endpoint import query_sagemaker_endpoint, query_hugging_face_endpoint
from .sagemaker_helpers.search_sagemaker_jumpstart_models import search_sagemaker_jumpstart_model
from .utils.model_utils import is_sagemaker_model
from .utils.rich_utils import print_model, print_error, print_success
from enum import StrEnum
from rich import print


class Actions(StrEnum):
    LIST = "Show active model endpoints"
    HUGGING_FACE = "Deploy a Hugging Face model"
    SAGEMAKER = "Deploy a SageMaker model"
    DELETE = "Delete a model endpoint"
    QUERY = "Query a model endpoint"
    EXIT = "Quit"
    # TRAIN = "fine tune a model"


def main(args, loglevel):
    logging.basicConfig(format="%(levelname)s: %(message)s", level=loglevel)
    session = boto3.session.Session()
    sagemaker_session = sagemaker.session.Session(boto_session=session)

    if args.hf is not None:
        # TODO: take args
        instance_type = select_instance()
        instance_count = 1
        predictor = deploy_huggingface_model(
            args.hf, instance_type, instance_count)
    while True:
        active_endpoints = list_sagemaker_endpoints()

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
            case Actions.SAGEMAKER:
                models = []
                while len(models) == 0:
                    models = search_sagemaker_jumpstart_model()
                    if models is None:
                        quit()

                questions = [
                    inquirer.List(
                        'model_id',
                        message="Choose a model",
                        choices=[model for model in models]
                    )
                ]
                answers = inquirer.prompt(questions)
                if (answers is None):
                    continue

                model_id, model_version = answers['model_id'], "2.*"

                # TODO: take args
                instance_type = select_instance()
                instance_count = 1
                predictor = create_and_deploy_jumpstart_model(
                    model_id, model_version, instance_type, instance_count)
                if predictor is None:
                    continue
                print_success(
                    f"{model_id} is now up and running at the endpoint [blue]{predictor.endpoint_name}")
            case Actions.HUGGING_FACE:
                questions = [
                    inquirer.Text(
                        'model_id',
                        message="Enter the exact model name from huggingface.co (e.g. google-bert/bert-base-uncased)",
                    )
                ]
                answers = inquirer.prompt(questions)
                if answers is None:
                    continue

                model_id = answers['model_id']

                # TODO: take args
                instance_type = select_instance()
                instance_count = 1
                predictor = deploy_huggingface_model(
                    model_id, instance_type, instance_count)

                if predictor is None:
                    continue

                print_success(
                    f"{model_id} is now up and running at the endpoint [blue]{predictor.endpoint_name} and is ready to be queried!")
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
                    inquirer.List('endpoint',
                                  message="Which endpoint would you like to query?",
                                  choices=[endpoint['EndpointName']
                                           for endpoint in active_endpoints]
                                  ),
                    inquirer.Text(
                        name='query', message='What would you like to query?')
                ]
                answers = inquirer.prompt(questions)
                if (answers is None):
                    continue

                endpoint = answers['endpoint']
                query = answers['query']
                if is_sagemaker_model(endpoint):
                    query_sagemaker_endpoint(endpoint, query)
                else:
                    query_hugging_face_endpoint(
                        sagemaker_session, endpoint, query)
            case Actions.EXIT:
                quit()

    print_success("Goodbye!")
