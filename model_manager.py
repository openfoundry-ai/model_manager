import argparse
import logging
import traceback
logging.getLogger("sagemaker.config").setLevel(logging.WARNING)
logging.getLogger("botocore.credentials").setLevel(logging.WARNING)
import os
from yaml import load, CSafeLoader
from pathlib import Path
from src.sagemaker_helpers.create_sagemaker_model import deploy_huggingface_model, deploy_model_config
from src.schemas import DeploymentConfig, ModelConfig


if __name__ == '__main__':
    # Run setup if these files/directories don't already exist
    if (not os.path.exists(os.path.expanduser('~/.aws')) or not os.path.exists('.env')):
        os.system("bash setup.sh")

    parser = argparse.ArgumentParser(
        description="Create, deploy, query against models.",
        epilog="As an alternative to the commandline, params can be placed in a file, one per line, and specified on the commandline like '%(prog)s @params.conf'.",
        fromfile_prefix_chars='@')
    parser.add_argument(
        "--hf",
        help="Deploy a Hugging Face Model.",
        type=str
    )
    parser.add_argument(
        "--instance",
        help="EC2 instance type to deploy to.",
        type=str
    )
    parser.add_argument(
        "--config",
        help="path to YAML configuration file",
        type=str
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="increase output verbosity",
        action="store_true")
    args = parser.parse_args()

    # Setup logging
    if args.verbose:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO

    if args.hf is not None:
        instance_type = args.instance or "ml.m5.xlarge"
        predictor = deploy_huggingface_model(args.hf, instance_type)
        quit()

    if args.config is not None:
        try:
            deployment_config = None
            model_config = None
            with open(args.config) as config:
                configuration = load(config, CSafeLoader)
                deployment_config = DeploymentConfig(
                    **configuration['deployment'])
                model_config = ModelConfig(**configuration['model'])
            deploy_model_config(deployment_config, model_config)
        except:
            traceback.print_exc()
            print("File not found")

        quit()

    from src.main import main
    main(args, loglevel)
