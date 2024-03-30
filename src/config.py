import glob
import yaml
from src.yaml import dumper
from src.schemas.model import Model
from src.schemas.deployment import Deployment


def get_configs():
    configurations = glob.glob("./configs/*.yaml")
    configs = {}

    for configuration in configurations:
        with open(configuration) as config:
            configuration = yaml.safe_load(config)
            deployment = configuration['deployment']
            # TODO: Support multi-model deployment
            model = configuration['models'][0]
            configs[deployment.endpoint_name] = (deployment, model)

    return configs


def write_config(deployment: Deployment, model: Model):
    with open(f"./configs/{deployment.endpoint_name}.yaml", 'w') as config:
        out = {
            "deployment": deployment,
            "models": [model],
        }
        config.write(yaml.dump(out, Dumper=dumper))
    return
