import inquirer
from enum import StrEnum, auto
from sagemaker.jumpstart.notebook_utils import list_jumpstart_models
from src.utils.rich_utils import print_error
from src.session import get_boto_session, get_sagemaker_session


class Frameworks(StrEnum):
    huggingface = auto()
    meta = auto()
    model = auto()
    tensorflow = auto()
    pytorch = auto()
    # autogluon
    # catboost
    # lightgbm
    mxnet = auto()
    # sklearn
    # xgboost


def search_sagemaker_jumpstart_model():
    questions = [
        inquirer.List('framework',
                      message="Which framework would you like to use?",
                      choices=[framework.value for framework in Frameworks]
                      ),
    ]
    answers = inquirer.prompt(questions)
    if answers is None:
        return
    filter_value = "framework == {}".format(answers["framework"])

    models = list_jumpstart_models(filter=filter_value,
                                   region=get_boto_session().region_name, sagemaker_session=get_sagemaker_session())
    return models
