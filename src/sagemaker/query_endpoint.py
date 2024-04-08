import boto3
import json
import inquirer
from InquirerPy import prompt
from sagemaker.huggingface.model import HuggingFacePredictor
from src.console import console
from src.sagemaker import SagemakerTask
from src.huggingface import HuggingFaceTask
from src.utils.model_utils import get_model_and_task, is_sagemaker_model, get_text_generation_hyperpameters
from src.utils.rich_utils import print_error
from src.schemas.deployment import Deployment
from src.schemas.model import Model
from src.schemas.query import Query
from src.session import sagemaker_session
from typing import Dict, Tuple, Optional


def make_query_request(endpoint_name: str, query: Query, config: Tuple[Deployment, Model]):
    if is_sagemaker_model(endpoint_name, config):
        return query_sagemaker_endpoint(endpoint_name, query, config)
    else:
        return query_hugging_face_endpoint(endpoint_name, query, config)


def parse_response(query_response):
    model_predictions = json.loads(query_response['Body'].read())
    probabilities, labels, predicted_label = model_predictions[
        'probabilities'], model_predictions['labels'], model_predictions['predicted_label']
    return probabilities, labels, predicted_label


def query_hugging_face_endpoint(endpoint_name: str, user_query: Query, config: Tuple[Deployment, Model]):
    task = get_model_and_task(endpoint_name, config)['task']
    predictor = HuggingFacePredictor(endpoint_name=endpoint_name,
                                     sagemaker_session=sagemaker_session)

    query = user_query.query
    context = user_query.context

    input = {"inputs": query}
    if task is not None and task == HuggingFaceTask.QuestionAnswering:
        if context is None:
            questions = [{
                "type": "input", "message": "What context would you like to provide?:", "name": "context"}]
            answers = prompt(questions)
            context = answers.get('context', '')

        if not context:
            raise Exception("Must provide context for question-answering")

        input = {}
        input['context'] = answers['context']
        input['question'] = query

    if task is not None and task == HuggingFaceTask.TextGeneration:
        parameters = get_text_generation_hyperpameters(config, user_query)
        input['parameters'] = parameters

    if task is not None and task == HuggingFaceTask.ZeroShotClassification:
        if context is None:
            questions = [
                inquirer.Text('labels',
                              message="What labels would you like to use? (comma separated values)?",
                              )
            ]
            answers = inquirer.prompt(questions)
            context = answers.get('labels', '')

        if not context:
            raise Exception(
                "Must provide labels for zero shot text classification")

        labels = context.split(',')
        input = json.dumps({
            "sequences": query,
            "candidate_labels": labels
        })

    try:
        result = predictor.predict(input)
    except Exception:
        console.print_exception()
        quit()

    print(result)
    return result


def query_sagemaker_endpoint(endpoint_name: str, user_query: Query, config: Tuple[Deployment, Model]):
    client = boto3.client('runtime.sagemaker')
    task = get_model_and_task(endpoint_name, config)['task']

    if task not in [
        SagemakerTask.ExtractiveQuestionAnswering,
        SagemakerTask.TextClassification,
        SagemakerTask.SentenceSimilarity,
        SagemakerTask.SentencePairClassification,
        SagemakerTask.Summarization,
        SagemakerTask.NamedEntityRecognition,
        SagemakerTask.TextEmbedding,
        SagemakerTask.TcEmbedding,
        SagemakerTask.TextGeneration,
        SagemakerTask.TextGeneration1,
        SagemakerTask.TextGeneration2,
        SagemakerTask.Translation,
        SagemakerTask.FillMask,
        SagemakerTask.ZeroShotTextClassification
    ]:
        print_error("""
Querying this model type inside of Model Manager isn’t yet supported. 
You can query it directly through the API endpoint - see here for documentation on how to do this:
https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_runtime_InvokeEndpoint.html
                    """)
        raise Exception("Unsupported")

    # MIME content type varies per deployment
    content_type = "application/x-text"
    accept_type = "application/json;verbose"

    # Depending on the task, input needs to be formatted differently.
    # e.g. question-answering needs to have {question: , context: }
    query = user_query.query
    context = user_query.context
    input = query.encode("utf-8")
    match task:
        case SagemakerTask.ExtractiveQuestionAnswering:
            if context is None:
                questions = [
                    {
                        'type': 'input',
                        'name': 'context',
                        'message': "What context would you like to provide?",
                    }
                ]
                answers = prompt(questions)
                context = answers.get("context", '')

            if not context:
                raise Exception("Must provide context for question-answering")

            content_type = "application/list-text"
            input = json.dumps([query, context]).encode("utf-8")

        case SagemakerTask.SentencePairClassification:
            if context is None:
                questions = [
                    inquirer.Text('context',
                                  message="What sentence would you like to compare against?",
                                  )
                ]
                answers = inquirer.prompt(questions)
                context = answers.get("context", '')
            if not context:
                raise Exception(
                    "Must provide a second sentence for sentence pair classification")

            content_type = "application/list-text"
            input = json.dumps([query, context]).encode("utf-8")
        case SagemakerTask.ZeroShotTextClassification:
            if context is None:
                questions = [
                    inquirer.Text('labels',
                                  message="What labels would you like to use? (comma separated values)?",
                                  )
                ]
                answers = inquirer.prompt(questions)
                context = answers.get('labels', '')

            if not context:
                raise Exception(
                    "must provide labels for zero shot text classification")
            labels = context.split(',')

            content_type = "application/json"
            input = json.dumps({
                "sequences": query,
                "candidate_labels": labels,
            }).encode("utf-8")
        case SagemakerTask.TextGeneration:
            parameters = get_text_generation_hyperpameters(config, user_query)
            input = json.dumps({
                "inputs": query,
                "parameters": parameters,
            }).encode("utf-8")
            content_type = "application/json"

    try:
        response = client.invoke_endpoint(
            EndpointName=endpoint_name, ContentType=content_type, Body=input, Accept=accept_type)
    except Exception:
        console.print_exception()
        quit()

    model_predictions = json.loads(response['Body'].read())
    print(model_predictions)
    return model_predictions


def test(endpoint_name: str):
    text1 = 'astonishing ... ( frames ) profound ethical and philosophical questions in the form of dazzling pop entertainment'
    text2 = 'simply stupid , irrelevant and deeply , truly , bottomlessly cynical '

    for text in [text1, text2]:
        query_sagemaker_endpoint(endpoint_name, text.encode('utf-8'))
