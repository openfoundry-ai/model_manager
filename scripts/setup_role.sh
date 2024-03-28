#!/usr/bin/env bash

# This script creates a role named SageMakerRole
# that can be used by SageMaker and has access to S3.

ROLE_NAME=SageMakerRole

# Creates a AWS policy for full access to sagemaker
POLICY=arn:aws:iam::aws:policy/AmazonSageMakerFullAccess

if `aws iam get-role --role-name ${ROLE_NAME} &> /dev/null` ; then 
	echo "SAGEMAKER_ROLE=`aws iam get-role --role-name ${ROLE_NAME} | grep -Eo '"arn:aws:iam.*?[^\\]"'`" >> .env
	exit
fi

cat <<EOF > /tmp/assume-role-policy-document.json
{
	"Version": "2012-10-17",
	"Statement": [{
		"Effect": "Allow",
		"Principal": {
			"Service": "sagemaker.amazonaws.com"
		},
		"Action": "sts:AssumeRole"
	}]
}
EOF

# Creates the role
echo "SAGEMAKER_ROLE=`aws iam create-role --role-name ${ROLE_NAME} --assume-role-policy-document file:///tmp/assume-role-policy-document.json | grep -Eo '"arn:aws:iam.*?[^\\]"'`" >> .env  

# attaches the Sagemaker full access policy to the role
aws iam attach-role-policy --policy-arn ${POLICY}  --role-name ${ROLE_NAME}