import inquirer
import logging
import threading
logging.getLogger("sagemaker.config").setLevel(logging.WARNING)
logging.getLogger("botocore.credentials").setLevel(logging.WARNING)
from InquirerPy import prompt
from .sagemaker_helpers.create_sagemaker_model import create_and_deploy_jumpstart_model, deploy_huggingface_model
from .sagemaker_helpers.delete_sagemaker_model import delete_sagemaker_model
from .sagemaker_helpers.sagemaker_resources import list_sagemaker_endpoints, select_instance, list_service_quotas_async
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

    print("[magenta]Model Manager by OpenFoundry.")
    print("[magenta]Star us on Github ☆! [blue]https://github.com/openfoundry-ai/model_manager")

    # list_service_quotas is a pretty slow API and it's paginated. Use async here and store the result in instances
    instances = []
    thr = threading.Thread(target=list_service_quotas_async, args=[instances])
    thr.start()

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
                thr.join()
                instance_type = select_instance(instances)
                predictor = create_and_deploy_jumpstart_model(
                    model_id, model_version, instance_type)
                if predictor is None:
                    continue
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
                thr.join()
                instance_type = select_instance(instances)
                predictor = deploy_huggingface_model(model_id, instance_type)

                if predictor is None:
                    continue

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
                query = answers['query']
                if is_sagemaker_model(endpoint):
                    query_sagemaker_endpoint(endpoint, query)
                else:
                    query_hugging_face_endpoint(endpoint, query)
            case Actions.EXIT:
                quit()

    print_success("Goodbye!")
