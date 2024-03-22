import boto3
import sagemaker
from sagemaker.session import Session

session = boto3.session.Session()
sagemaker_session = sagemaker.session.Session(boto_session=session)
