import boto3
from functools import lru_cache
from InquirerPy import inquirer
from src.console import console
from src.sagemaker_helpers import EC2Instance
from src.config import get_config_for_endpoint
from src.utils.format import format_sagemaker_endpoint, format_python_dict
from typing import List, Tuple, Dict, Optional


def list_sagemaker_endpoints(filter_str: str = None) -> List[str]:
    sagemaker_client = boto3.client('sagemaker')

    endpoints = sagemaker_client.list_endpoints()['Endpoints']
    if filter_str is not None:
        endpoints = list(filter(lambda x: filter_str ==
                         x['EndpointName'], endpoints))

    for endpoint in endpoints:
        endpoint_config = sagemaker_client.describe_endpoint_config(
            EndpointConfigName=endpoint['EndpointName'])['ProductionVariants'][0]
        endpoint['InstanceType'] = endpoint_config['InstanceType']
    return endpoints


def get_sagemaker_endpoint(endpoint_name: str) -> Optional[Dict[str, Optional[Dict]]]:
    endpoints = list_sagemaker_endpoints(endpoint_name)
    if not endpoints:
        return None

    endpoint = format_sagemaker_endpoint(endpoints[0])

    config = get_config_for_endpoint(endpoint_name)
    if config is None:
        return {'deployment': endpoint, 'model': None}

    deployment, model = config
    deployment = format_python_dict(deployment.model_dump())
    model = format_python_dict(model.model_dump())

    # Merge the endpoint dict with our config
    deployment = {**endpoint, **deployment}

    return {
        'deployment': deployment,
        'model': model,
    }


@lru_cache
def list_service_quotas() -> List[Tuple[str, int]]:
    """ Gets a list of EC2 instances for inference """

    client = boto3.client('service-quotas')
    quotas = []
    try:
        response = client.list_service_quotas(
            ServiceCode="sagemaker",
            MaxResults=100,
        )
        next_token = response.get('NextToken')
        quotas = response['Quotas']
        while next_token is not None:
            response = client.list_service_quotas(
                ServiceCode="sagemaker",
                NextToken=next_token,
            )
            quotas.extend(response['Quotas'])
            next_token = response.get('NextToken')
    except client.exceptions.AccessDeniedException as error:
        console.print(
            "[red]User does not have access to Service Quotas. Grant access via IAM to get the list of available instances")
        return []

    # TODO: Filter for different usages
    available_instances = list(filter(lambda x: 'endpoint usage' in x[0] and x[1] > 0, [
                               (quota['QuotaName'], quota['Value']) for quota in quotas]))

    # Clean up quota names
    available_instances = [(instance[0].split(" ")[0], instance[1])
                           for instance in available_instances]
    return available_instances


def list_service_quotas_async(instances=[]):
    """ Wrapper to allow access to list in threading """
    instances = instances.extend(list_service_quotas())
    return instances


def select_instance(available_instances=None):
    choices = [instance[0] for instance in available_instances] or [
        instance for instance in EC2Instance]
    instance = inquirer.fuzzy(
        message="Choose an instance size (note: ml.m5.xlarge available by default; you must request quota from AWS to use other instance types):",
        choices=choices,
        default="ml.m5."
    ).execute()
    if instance is None:
        return
    return instance
