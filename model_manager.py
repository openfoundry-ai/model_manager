import argparse
import logging
import traceback
import yaml
logging.getLogger("sagemaker.config").setLevel(logging.WARNING)
logging.getLogger("botocore.credentials").setLevel(logging.WARNING)
import os
from src.sagemaker.create_model import deploy_huggingface_model, deploy_model
from src.schemas.deployment import Deployment
from src.schemas.model import Model


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
            deployment = None
            model = None
            with open(args.config) as config:
                configuration = yaml.safe_load(config)
                deployment = configuration['deployment']

                # TODO: Support multi-model endpoints
                model = configuration['models'][0]
            deploy_model(deployment, model)
        except:
            traceback.print_exc()
            print("File not found")

        quit()

    from src.main import main
    main(args, loglevel)
