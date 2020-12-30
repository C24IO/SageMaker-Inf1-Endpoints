# Source - https://aws.amazon.com/blogs/machine-learning/fine-tuning-a-pytorch-bert-model-and-deploying-it-with-amazon-elastic-inference-on-amazon-sagemaker/

import os

import pandas as pd
import sagemaker
from sklearn.model_selection import train_test_split

sagemaker_session = sagemaker.Session()

bucket = sagemaker_session.default_bucket()
prefix = "sagemaker/DEMO-pytorch-bert"

# role = sagemaker.get_execution_role()
role = 'arn:aws:iam::111652037296:role/service-role/AmazonSageMaker-ExecutionRole-20180406T143398'


def exec_cmd(cmd):
    f = os.popen(cmd)
    now = f.read()
    print(now)


if not os.path.exists("./cola_public_1.1.zip"):
    exec_cmd('curl -o ./cola_public_1.1.zip https://nyu-mll.github.io/CoLA/cola_public_1.1.zip')
if not os.path.exists("./cola_public/"):
    exec_cmd('unzip cola_public_1.1.zip')

df = pd.read_csv(
    "./cola_public/raw/in_domain_train.tsv",
    sep="\t",
    header=None,
    usecols=[1, 3],
    names=["label", "sentence"],
)
sentences = df.sentence.values
labels = df.label.values

print(sentences[20:25])
print(labels[20:25])



train, test = train_test_split(df)
train.to_csv("./cola_public/train.csv", index=False)
test.to_csv("./cola_public/test.csv", index=False)

inputs_train = sagemaker_session.upload_data("./cola_public/train.csv", bucket=bucket, key_prefix=prefix)
inputs_test = sagemaker_session.upload_data("./cola_public/test.csv", bucket=bucket, key_prefix=prefix)

from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point="train_deploy.py",
    source_dir="code",
    role=role,
    framework_version="1.5.0",
    py_version="py3",
    instance_count=2,  # this script only support distributed training for GPU instances.
    instance_type="ml.p3.2xlarge",
    hyperparameters={
        "epochs": 1,
        "num_labels": 2,
        "backend": "gloo",
    }
)
# estimator.fit({"training": inputs_train, "testing": inputs_test})

# predictor = estimator.deploy(initial_instance_count=1, instance_type="ml.m4.xlarge")

# from sagemaker.predictor import json_deserializer, json_serializer

# predictor.content_type = "application/json"
# predictor.accept = "application/json"
# predictor.serializer = sagemaker.serializers.JSONSerializer()
# predictor.deserializer = sagemaker.deserializers.JSONDeserializer()

# result = predictor.predict("Somebody just left - guess who.")
# print(np.argmax(result, axis=1))

# predictor.delete_endpoint()

# Pretrained model

from sagemaker.pytorch.model import PyTorchModel

pytorch_model = PyTorchModel(
    model_data="s3://sagemaker-us-east-1-111652037296/pytorch-training-2020-12-29-19-35-32-544/output/model.tar.gz",
    role=role,
    framework_version="1.3.1",
    source_dir="code",
    py_version="py3",
    entry_point="train_deploy.py")

##predictor = pytorch_model.deploy(initial_instance_count=1, instance_type="ml.m4.xlarge")

# print(estimator.model_data)

import subprocess
import torch
from transformers import BertForSequenceClassification

model_torchScript = BertForSequenceClassification.from_pretrained("model/", torchscript=True)
device = "cpu"
for_jit_trace_input_ids = [0] * 64
for_jit_trace_attention_masks = [0] * 64
for_jit_trace_input = torch.tensor([for_jit_trace_input_ids])
for_jit_trace_masks = torch.tensor([for_jit_trace_input_ids])

traced_model = torch.jit.trace(
    model_torchScript, [for_jit_trace_input.to(device), for_jit_trace_masks.to(device)]
)
torch.jit.save(traced_model, "traced_bert.pt")

subprocess.call(["tar", "-czvf", "traced_bert.tar.gz", "traced_bert.pt"])

from sagemaker.pytorch import PyTorchModel

instance_type = 'ml.m5.large'
accelerator_type = 'ml.eia2.xlarge'

# TorchScript model
tar_filename = 'traced_bert.tar.gz'

# Returns S3 bucket URL
print('Upload tarball to S3')
model_data = sagemaker_session.upload_data(path=tar_filename, bucket=bucket, key_prefix=prefix)

endpoint_name = 'bert-ei-traced-{}-{}'.format(instance_type, accelerator_type).replace('.', '').replace('_', '')

pytorch = PyTorchModel(
    model_data=model_data,
    role=role,
    entry_point='deploy_ei.py',
    source_dir='code',
    framework_version='1.3.1',
    py_version='py3',
    sagemaker_session=sagemaker_session
)

# Function will exit before endpoint is finished creating
predictor = pytorch.deploy(
    initial_instance_count=1,
    instance_type=instance_type,
    accelerator_type=accelerator_type,
    endpoint_name=endpoint_name,
    wait=True
)
