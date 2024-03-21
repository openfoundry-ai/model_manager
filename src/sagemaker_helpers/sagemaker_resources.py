import boto3
import inquirer
from typing import List
from rich import print
from enum import StrEnum
from . import EC2Instance


def list_sagemaker_endpoints(filter_str: str = None) -> List[str]:
    sagemaker_client = boto3.client('sagemaker')

    endpoints = sagemaker_client.list_endpoints()['Endpoints']

    if filter_str is not None:
        endpoints = list(filter(lambda x: filter_str in x, endpoints))

    for endpoint in endpoints:
        endpoint_config = sagemaker_client.describe_endpoint_config(
            EndpointConfigName=endpoint['EndpointName'])['ProductionVariants'][0]
        endpoint['InstanceType'] = endpoint_config['InstanceType']
    return endpoints


def list_service_quotas():
    client = boto3.client('service-quotas')
    response = client.list_service_quotas(
        ServiceCode="sagemaker"
    )
    print(response)


def select_instance() -> EC2Instance:
    questions = [
        inquirer.List(
            'instance',
            message="Choose an instance size (note: ml.m5.xlarge available by default; you must request quota from AWS to use other instance types)",
            choices=[instance for instance in EC2Instance]
        )
    ]
    answers = inquirer.prompt(questions)
    if answers is None:
        return
    return answers['instance']
