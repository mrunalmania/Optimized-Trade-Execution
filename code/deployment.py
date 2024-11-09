#!/usr/bin/env python
import boto3
import sagemaker
from sagemaker.pytorch import PyTorchModel
from sagemaker import get_execution_role

# Set up the role and client
role = "arn:aws:iam::221260211166:role/sagemakerrole"
sagemaker_client = boto3.client('sagemaker')

# Define the S3 path to your model
model_data = 's3://test-bucket-zybook/dqn_model.tar.gz'

# Set up the PyTorchModel for SageMaker
pytorch_model = PyTorchModel(
    model_data=model_data,
    role=role,
    image_uri="221260211166.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:latest",   # Your inference script
    framework_version='1.0.0',
    source_dir = '/opt/ml/model',
    py_version='py3',
)

# Deploy the model as a real-time endpoint
predictor = pytorch_model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large'
)

# Save the endpoint name for future reference
endpoint_name = predictor.endpoint_name
print("Model deployed to SageMaker endpoint:", endpoint_name)
