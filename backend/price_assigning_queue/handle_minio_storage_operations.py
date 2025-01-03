import os
from minio import Minio
import datetime

from PIL import Image
import requests
from io import BytesIO

def create_minio_client():
    client = Minio(
        endpoint=os.getenv('MINIO_HOST', 'localhost') + ':' + os.getenv('MINIO_PORT', '9000'),
        access_key=os.getenv('MINIO_ROOT_USER', 'minioadmin'),
        secret_key=os.getenv('MINIO_ROOT_PASSWORD', 'minioadmin'),
        secure=False
    )
    return client


def generate_presigned_url(client, file_name, bucket_name, expiry=24*60*60):
    try:
        # Ensure the bucket exists before attempting to generate the URL
        if not client.bucket_exists(bucket_name):
            raise ValueError(f"Bucket '{bucket_name}' does not exist.")
        
        # Generate the pre-signed download URL
        presigned_url = client.presigned_get_object(bucket_name, file_name, expires=datetime.timedelta(seconds=expiry))
        return presigned_url
    except Exception as e:
        print(f"Error generating pre-signed download URL: {e}")
        return {"error": "Failed to generate pre-signed download URL"}
    

def fetch_images_from_presigned_urls(urls):
    images = []
    for url in urls:
        # print("Fetching image from URL: ", url)
        response = requests.get(url)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        images.append(image)
    return images