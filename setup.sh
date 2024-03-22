#!/bin/sh
python3 -m ensurepip --upgrade
make
source venv/bin/activate
if ! command -v aws &> /dev/null
then
    OS="$(uname -s)"
    case "${OS}" in
        Linux*)     
            curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
            unzip awscliv2.zip
            sudo ./aws/install
            ;;
        Darwin*)    
            curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"
            sudo installer -pkg AWSCLIV2.pkg -target /
            ;;
        *)          
            echo "Unsupported OS: ${OS}. See https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
            exit 1
            ;;
    esac
fi
aws configure set region us-east-1 && aws configure
touch .env
if ! grep -q "SAGEMAKER_ROLE" .env
then
    bash ./scripts/setup_role.sh
fi
source venv/bin/activate