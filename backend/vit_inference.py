
import torch

from matplotlib import pyplot as plt
from model.google_vit_model import get_vit_model
from helpers.data_loader import get_data_loaders

from helpers.processed_data import processed_data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

images, prices = processed_data(100)

model, feature_extractor = get_vit_model(aggregation_method="mean")

model.load_state_dict(torch.load("models/vit_regression_model.pth"))

model.to(device)

train_loader, val_loader = get_data_loaders(images, prices, feature_extractor)


def show_predictions(val_loader, model, device):
    model.eval()
    with torch.no_grad():
        for pixel_values_list, prices in val_loader:
            outputs = model([instance.to(device) for instance in pixel_values_list])
            for batch_idx, (batch, predicted, actual) in enumerate(zip(pixel_values_list, outputs, prices)):
                n_images = len(batch)  # Get the number of images in the batch
                plt.figure(figsize=(n_images * 5, 5))  # Adjust figure size based on number of images
                for idx, img_tensor in enumerate(batch):
                    plt.subplot(1, n_images, idx + 1)
                    img_tensor = img_tensor.cpu().squeeze()
                    if img_tensor.dim() == 3:  # Check if the image has three dimensions (C, H, W)
                        img_tensor = img_tensor.permute(1, 2, 0)  # Permute to (H, W, C) for imshow
                    plt.imshow(img_tensor.numpy())
                    plt.axis('off')  # Turn off axis numbers and ticks
                plt.suptitle(f"Sample {batch_idx + 1}: Predicted: ${predicted.item():.2f}, Actual: ${actual.item():.2f}", fontsize=16)
                plt.show()

# Call the function to show predictions after the last epoch
show_predictions(val_loader, model, device)
