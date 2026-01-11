# scripts/init_minio.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

# Настройки из .env
S3_ENDPOINT_URL = "http://localhost:9000"
S3_ACCESS_KEY = "minioadmin"
S3_SECRET_KEY = "minioadmin"
S3_BUCKET_NAME = "face-recognition"
S3_REGION = "us-east-1"

def init_minio():
    print("🚀 Initializing MinIO...")
    
    s3_client = boto3.client(
        's3',
        endpoint_url=S3_ENDPOINT_URL,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        config=Config(signature_version='s3v4'),
        region_name=S3_REGION,
        verify=False
    )
    
    try:
        s3_client.head_bucket(Bucket=S3_BUCKET_NAME)
        print(f"✅ Bucket '{S3_BUCKET_NAME}' already exists")
    except ClientError:
        try:
            s3_client.create_bucket(Bucket=S3_BUCKET_NAME)
            print(f"✅ Created bucket '{S3_BUCKET_NAME}'")
        except Exception as e:
            print(f"❌ Failed to create bucket: {e}")
            return False
    
    try:
        cors_config = {
            'CORSRules': [{
                'AllowedHeaders': ['*'],
                'AllowedMethods': ['GET', 'PUT', 'POST', 'DELETE'],
                'AllowedOrigins': ['*'],
                'ExposeHeaders': ['ETag']
            }]
        }
        s3_client.put_bucket_cors(Bucket=S3_BUCKET_NAME, CORSConfiguration=cors_config)
        print("✅ CORS configured")
    except Exception as e:
        print(f"⚠️  Warning: {e}")
    
    print("✨ MinIO initialization complete!")
    return True

if __name__ == "__main__":
    success = init_minio()
    sys.exit(0 if success else 1)
