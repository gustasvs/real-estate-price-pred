import matplotlib.pyplot as plt
import torchvision.transforms as transforms

# Add this function to convert tensor to image
def show_image_from_tensor(tensor):
    """
    Display an image from the tensor passed to the model.
    Args:
        tensor (torch.Tensor): Tensor with shape [3, 224, 224] (after preprocessing).
    """
    # Convert tensor to PIL image
    inv_normalize = transforms.Normalize(
        mean=[-0.5, -0.5, -0.5],
        std=[1/0.5, 1/0.5, 1/0.5]
    )
    tensor = inv_normalize(tensor)  # Revert normalization
    image = transforms.ToPILImage()(tensor.cpu()).convert("RGB")

    # Plot image
    plt.imshow(image)
    plt.axis("off")
    plt.show()
