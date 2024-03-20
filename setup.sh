#!/bin/sh
python3 -m ensurepip --upgrade
make
if ! command -v aws &> /dev/null
then
    curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"
    sudo installer -pkg ./AWSCLIV2.pkg -target /
fi
aws configure set region us-east-1 && aws configure
touch .env
if ! grep -q "SAGEMAKER_ROLE" .env
then
    bash ./scripts/setup_role.sh
fi
source venv/bin/activate