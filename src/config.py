import glob
import yaml
from src.yaml import dumper
from src.schemas.model import Model
from src.schemas.deployment import Deployment
from typing import Dict, Tuple


def get_configs() -> Dict[str, Tuple[Deployment, Model]]:
    configurations = glob.glob("./configs/*.yaml")
    configs = {}

    for configuration in configurations:
        with open(configuration) as config:
            configuration = yaml.safe_load(config)
            if configuration is None:
                continue

            deployment = configuration['deployment']
            # TODO: Support multi-model deployment
            model = configuration['models'][0]
            configs[deployment.endpoint_name] = (deployment, model)

    return configs


def get_config_for_endpoint(endpoint_name: str) -> Tuple[Deployment, Model]:
    return get_configs().get(endpoint_name)


def write_config(deployment: Deployment, model: Model):
    with open(f"./configs/{deployment.endpoint_name}.yaml", 'w') as config:
        out = {
            "deployment": deployment,
            "models": [model],
        }
        config.write(yaml.dump(out, Dumper=dumper))
    return
