import os
import boto3
import botocore
from boto.s3.key import Key

AWS_ACCESS_KEY_ID = 'AKIAICMGGEWQQEDSPKEA'
AWS_SECRET_ACCESS_KEY = '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'
BUCKET = 'google-colab-bucket'

def upload_to_s3(filename, key):
    file = open(filename, "rb")

    s3 = boto3.resource('s3',
                        aws_access_key_id=AWS_ACCESS_KEY_ID,
                        aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

    s3.Bucket(BUCKET).put_object(Key=key, Body=file)


def upload_folder_to_s3(local_folder):
    s3 = boto3.resource('s3',
                        aws_access_key_id=AWS_ACCESS_KEY_ID,
                        aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    file_names = os.listdir(local_folder)

    for file_name in file_names:
        file_name = os.path.join(local_folder, file_name)
        file = open(file_name, "rb")
        key = file_name
        s3.Bucket(BUCKET).put_object(Key=key, Body=file)


def download_from_s3(file, key):
    # Create an S3 client
    s3 = boto3.resource('s3',
                        aws_access_key_id=AWS_ACCESS_KEY_ID,
                        aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

    try:
        s3.Bucket(BUCKET).download_file(key, file)
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            raise


def download_folder_from_s3(local_folder, s3_folder):
    # Create an S3 client

    s3 = boto3.resource('s3',
                        aws_access_key_id=AWS_ACCESS_KEY_ID,
                        aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

    file_names = []
    for object in s3.Bucket(BUCKET).objects.all():
        if s3_folder in object.key and object.key[-1] != "/":
            s3_path = os.path.basename(object.key)
            if not os.path.exists(local_folder):
                os.makedirs(local_folder)
            s3.Bucket(BUCKET).download_file(object.key, os.path.join(local_folder, s3_path))