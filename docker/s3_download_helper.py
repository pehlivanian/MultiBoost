#!/usr/bin/env python3
"""
S3 download helper script for C++ executables.
Downloads dataset files from S3 when local files are not found.
"""

import json
import os
import sys
import tempfile
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from pathlib import Path


def download_s3_dataset(s3_config_file, dataset_name, output_dir):
    """
    Download dataset files from S3 to local directory.
    
    Args:
        s3_config_file: Path to JSON file containing S3 configuration
        dataset_name: Name of dataset (e.g., "synthetic_train")
        output_dir: Directory to download files to
    
    Returns:
        0 on success, 1 on error
    """
    try:
        # Load S3 configuration
        with open(s3_config_file, 'r') as f:
            s3_config = json.load(f)
        
        # Extract S3 configuration
        bucket = s3_config['bucket']
        prefix = s3_config.get('prefix', '').rstrip('/')
        access_key = s3_config['access_key']
        secret_key = s3_config['secret_key']
        region = s3_config.get('region', 'us-east-1')
        
        # Initialize S3 client
        s3_client = boto3.client(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region
        )
        
        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # File patterns to download
        file_patterns = [
            f"{dataset_name}_X.csv",
            f"{dataset_name}_y.csv"
        ]
        
        downloaded_files = []
        
        # Download each file that exists
        for pattern in file_patterns:
            s3_key = f"{prefix}/{pattern}" if prefix else pattern
            local_file_path = os.path.join(output_dir, pattern)
            
            try:
                s3_client.download_file(bucket, s3_key, local_file_path)
                downloaded_files.append(pattern)
                print(f"Downloaded {s3_key} to {local_file_path}", file=sys.stderr)
            except ClientError as e:
                if e.response['Error']['Code'] == '404':
                    # File doesn't exist, skip it
                    continue
                else:
                    raise
        
        if not downloaded_files:
            print(f"Error: No dataset files found in S3 bucket {bucket} with prefix {prefix}", file=sys.stderr)
            return 1
        
        print(f"Successfully downloaded {len(downloaded_files)} files: {downloaded_files}", file=sys.stderr)
        return 0
        
    except NoCredentialsError:
        print("Error: AWS credentials not provided or invalid", file=sys.stderr)
        return 1
    except ClientError as e:
        print(f"Error: S3 error: {str(e)}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: Failed to download S3 dataset: {str(e)}", file=sys.stderr)
        return 1


def main():
    if len(sys.argv) != 4:
        print("Usage: s3_download_helper.py <s3_config_file> <dataset_name> <output_dir>", file=sys.stderr)
        sys.exit(1)
    
    s3_config_file = sys.argv[1]
    dataset_name = sys.argv[2]
    output_dir = sys.argv[3]
    
    exit_code = download_s3_dataset(s3_config_file, dataset_name, output_dir)
    sys.exit(exit_code)


if __name__ == '__main__':
    main()