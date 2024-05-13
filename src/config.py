import glob
import yaml
from src.yaml import dumper
from src.schemas.model import Model
from src.schemas.deployment import Deployment
from typing import Dict, Tuple, List, NamedTuple, Optional


class ModelDeployment(NamedTuple):
    deployment: Deployment
    models: List[Model]


def get_deployment_configs(path: Optional[str] = None) -> List[ModelDeployment]:
    if path is None:
        path = "./configs/*.yaml"

    configurations = glob.glob(path)
    configs = []

    for configuration in configurations:
        with open(configuration) as config:
            configuration = yaml.safe_load(config)
            if configuration is None:
                continue

            # Filter out training configs
            if configuration.get('deployment') is None:
                continue

            deployment = configuration['deployment']
            models = configuration['models']
            configs.append(ModelDeployment(
                deployment=deployment, models=models))

    return configs


def get_endpoints_for_model(model_id: str, path: Optional[str] = None) -> List[ModelDeployment]:
    configs = get_deployment_configs(path)
    endpoints = []
    for config in configs:
        models = [model.id for model in config.models]
        if model_id in models:
            # TODO: Check if endpoint is still active
            endpoints.append(config)
    return endpoints


def get_config_for_endpoint(endpoint_name: str) -> Optional[ModelDeployment]:
    configs = get_deployment_configs()
    for config in configs:
        if config.deployment.endpoint_name == endpoint_name:
            return config
    return None


def write_config(deployment: Deployment, model: Model):
    with open(f"./configs/{deployment.endpoint_name}.yaml", 'w') as config:
        out = {
            "deployment": deployment,
            "models": [model],
        }
        config.write(yaml.dump(out, Dumper=dumper))
    return
