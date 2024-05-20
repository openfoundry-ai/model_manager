import boto3
import boto3.session
import sagemaker
from src.supabase import supabase_client
from src.supabase.secret import get_secrets
from src.schemas.secret import SecretKeys


def get_boto_session():
    if supabase_client is not None and supabase_client.auth.get_user() is not None:
        user_secrets = get_secrets(supabase_client.auth.get_user().user.id)
        aws_access_key = next((secret.value.get_secret_value(
        ) for secret in user_secrets if secret.key == SecretKeys.AWSAccessKey), None).decode()
        aws_secret_access_key = next((secret.value.get_secret_value(
        ) for secret in user_secrets if secret.key == SecretKeys.AWSSecretAccessKey), None).decode()

        # Sort out region
        return boto3.session.Session(
            aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_access_key, region_name='us-east-1')
    else:
        # rely on local setup
        return boto3.session.Session()


def get_sagemaker_session():
    session = get_boto_session()
    return sagemaker.session.Session(boto_session=session)
