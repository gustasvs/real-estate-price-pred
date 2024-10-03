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
    processed_image_folder = "data/processed_dataset"
    images = []
    for i in range(1, count):
        image_path = f"{image_folder}/{i}_bathroom.jpg"
        image = mpimg.imread(image_path)
        images.append(image)

        # print("*" * 20)
        # print(f"Image shape: {image.shape}")
        # print(f"Image size: {image.size}")
        # print(f"Image type: {image.dtype}")
    
        # plt.imshow(image)
        # plt.show()

    return images
    

if __name__ == "__main__":
    images = load_images(5)
    