import numpy as np

from load_images import load_images

def preprocess_sample_images(images, patch_size):
    patch_sequences = []
    boundary_token = np.zeros((1, patch_size, patch_size, 3))  # Example of a static boundary token
    
    for img in images:
        patches = extract_patches_from_image(img, patch_size)
        patch_sequences.extend(patches)
        patch_sequences.append(boundary_token)  # Add boundary token after each image's patches

    return np.array(patch_sequences)

def extract_patches_from_image(image, patch_size):
    # Extract patches of size `patch_size x patch_size` from the image
    # This is a placeholder for the actual patch extraction logic
    patches = []
    # Example patch extraction logic
    for i in range(0, image.shape[0], patch_size):
        for j in range(0, image.shape[1], patch_size):
            patch = image[i:i+patch_size, j:j+patch_size, :]
            patches.append(patch)
    return patches

def preprocess_images():
    # Test the preprocess_images function
    images = load_images(5)
    patch_size = 32
    patch_sequences = preprocess_sample_images(images, patch_size)
    print(f"Number of patches: {len(patch_sequences)}")
    print(f"Shape of patch sequences: {patch_sequences.shape}")

    return patch_sequences

if __name__ == "__main__":
    preprocess_images()
