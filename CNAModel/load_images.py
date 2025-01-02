from PIL import Image
import os
import random

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def load_images(count: int):
    """
    Load images from the file system.

    Args:
        count: The number of images to load.

    Returns:
        A list of image paths.
    """

    image_folder = "data/dataset"
    images = []
    
    for i in range(1, count + 1):
        sample_images = []
        categories = ["bathroom", "bedroom", "frontal", "kitchen"]

        for category in categories:
            image_path = f"{image_folder}/{i}_{category}.jpg"
            if os.path.exists(image_path):
                image = Image.open(image_path).convert("RGB")
                image = image.resize((224, 224))
                sample_images.append(image)
        
        images.append(sample_images)

    return images
    

if __name__ == "__main__":
    images = load_images(5)
    