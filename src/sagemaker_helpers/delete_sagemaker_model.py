import boto3
from rich import print
from src.utils.rich_utils import print_success
from typing import List


def delete_sagemaker_model(endpoint_names: List[str] = None):
    sagemaker_client = boto3.client('sagemaker')

    if len(endpoint_names) == 0:
        print_success("No Endpoints to delete!")
        return

    # Add validation / error handling
    for endpoint in endpoint_names:
        print(f"Deleting [blue]{endpoint}")
        sagemaker_client.delete_endpoint(EndpointName=endpoint)
