import matplotlib.pyplot as plt
from PIL import Image
from transformers import ViTImageProcessor
import torch

import os

# Specify the directory containing the images
image_dir = 'data_from_web/images/'

# Get all jpg image files from the directory
image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

# Initialize the image processorpp
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

for idx, filename in enumerate(image_files):
    image_path = os.path.join(image_dir, filename)
    image = Image.open(image_path)

    # Display the original image
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    # pl

    # Process the image
    processed = processor(image, return_tensors="pt")["pixel_values"][0]

    # Remove the batch dimension before applying other transformations
    processed = processed.squeeze(0).permute(1, 2, 0)  # Rearrange dimensions to [H, W, C]
    # plt.imshow(processed.clip(0, 1))  # Clip to maintain valid image pixel range

    # Display the processed image
    plt.subplot(1, 2, 2)
    plt.imshow(processed.clip(0, 1))
    plt.title('Processed Image')
    plt.show()
