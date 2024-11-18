import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam
from transformers import ViTImageProcessor
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import torchvision.transforms as T
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

import time

import transformers
transformers.logging.set_verbosity_error()


from load_images import load_images
from load_prices import load_prices

from config.settings import BATCH_SIZE, ENABLE_DEV_LOGS, LEARNING_RATE, LOSS_FUNCT, EPOCHS, BATCHES_TO_AGGREGATE

from model.google_vit_model import get_vit_model

from helpers.label_smothing import apply_lds, apply_fds, adaptive_lds

from helpers.weighted_loss import compute_bin_weights, WeightedMSELoss

model = get_vit_model()

# Define the custom dataset
class ImageDataset(Dataset):
    def __init__(self, images, prices, feature_extractor):
        self.images = images 
        self.prices = prices
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        sample_images = self.images[idx]

        sampled_count = np.random.randint(1, len(sample_images) + 1)
        sample_images = np.random.choice(sample_images, sampled_count, replace=False)

        price = self.prices[idx]
        
        # Preprocess and stack images
        sample_images_extracted = torch.stack([
            self.feature_extractor(images=img, return_tensors="pt")['pixel_values'].squeeze(0)
            for img in sample_images
        ])

        if ENABLE_DEV_LOGS: print("current sample shape: ", sample_images_extracted.shape)

        return sample_images_extracted, torch.tensor(price, dtype=torch.float32)


def train(model, dataloader, optimizer, device, epoch):
    model.train()
    running_loss = 0
    start_time = time.time()
    total_batches = len(dataloader)

    optimizer.zero_grad()

    for batch_idx, (sample, prices) in enumerate(dataloader, 1):

        sample = [instance.to(device) for instance in sample]

        if ENABLE_DEV_LOGS: print("one input in batch shape: ", sample[0].shape)  # Should print [8, 2, 3, 224, 224] for a batch size of 8

        prices = prices.to(device)

        if ENABLE_DEV_LOGS: print("prices shape (should be batch size): ", prices.shape)

        

        outputs = model(sample)
        loss = LOSS_FUNCT(outputs, prices) 
        loss.backward()

        if (batch_idx + 1) % BATCHES_TO_AGGREGATE == 0 or batch_idx == total_batches:
            optimizer.step()
            optimizer.zero_grad()

        # update running loss
        running_loss += loss.item()
        
        #  logs
        if batch_idx % 10 == 0 or batch_idx == total_batches:
            elapsed_time = time.time() - start_time
            avg_loss = running_loss / batch_idx
            print(
                f"Epoch [{epoch+1}] - Batch [{batch_idx}/{total_batches}]: "
                f"Batch Loss = {loss.item():.4f}, Avg Loss = {avg_loss:.4f}, "
                f"Elapsed Time = {elapsed_time:.2f}s"
            )
            # print last output and real outputs
            print("Last output: ", outputs[-1].item())
            print("Real output: ", prices[-1].item())

    return running_loss / total_batches


def validate(model, dataloader, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for pixel_values_list, prices in dataloader:
            pixel_values_list = [instance.to(device) for instance in pixel_values_list]

            prices = prices.to(device)
            
            outputs = model(pixel_values_list)
            loss = LOSS_FUNCT(outputs.squeeze(), prices)
            val_loss += loss.item()
    return val_loss / len(dataloader)

# Main script
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Batch size: {BATCH_SIZE}, aggregation multiplier: {BATCHES_TO_AGGREGATE}")
print("-" * 20)
print(f"Simulated batch size: {BATCH_SIZE * BATCHES_TO_AGGREGATE}")
print("-" * 20)

model.to(device)

count = 500
images = load_images(count) # shape = [count, image_count, image]
print("Images loaded...")

# prices = [np.random.randint(100, 1000) for _ in range(count)]
prices = load_prices(count) # shape = [count]
print("Prices loaded...")


sorted_prices = np.argsort(np.array(prices))
sorted_prices = [i for i in sorted_prices if prices[i] > 0]
print("Top 10 lowest prices: ", [prices[i] for i in sorted_prices[:10]])
print("Top 10 highest prices: ", [prices[i] for i in sorted_prices[-10:]])

filtered_indices = [i for i in range(len(prices)) if prices[i] <= 1_600_000]
filtered_prices = [prices[i] for i in filtered_indices]
filtered_images = [images[i] for i in filtered_indices]
prices = filtered_prices
images = filtered_images

count = len(prices)



scaler = MinMaxScaler()
prices = scaler.fit_transform(np.array(prices).reshape(-1, 1)).flatten()


# while True:
#     sigma = float(input("Enter the sigma value for LDS: "))
#     # density_threshold = float(input("Enter the density threshold for adaptive LDS: "))
#     # max_sigma = float(input("Enter the max sigma value for adaptive LDS: "))
#     # min_sigma = float(input("Enter the min sigma value for adaptive LDS: "))

#     fig, ax = plt.subplots(3, 1, figsize=(10, 10))
#     fig.canvas.manager.window.wm_geometry("+10+10")
    
#     ax[0].hist(prices, bins=50)
#     # ax[1].hist(apply_lds(prices, sigma=sigma), bins=50)
#     # ax[1].hist(adaptive_lds(prices, density_threshold=density_threshold, max_sigma=max_sigma, min_sigma=min_sigma), bins=50)
#     smoothed_prices = apply_fds(prices, prices, sigma=sigma)
#     ax[1].hist(smoothed_prices, bins=50)
#     ax[2].hist(apply_fds(smoothed_prices, smoothed_prices, sigma=sigma), bins=50)

#     plt.show()

# prices = apply_fds(prices, prices, sigma=1.05)

weights, weight_bins = compute_bin_weights(prices, num_bins=20)

fig, ax = plt.subplots(2, 1, figsize=(10, 5))
ax[0].hist(prices, bins=20)
ax[1].plot(weight_bins)
plt.show()

LOSS_FUNCT = WeightedMSELoss(weights)


feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
train_images, val_images, train_prices, val_prices = train_test_split(images, prices, test_size=0.2)
train_dataset = ImageDataset(train_images, train_prices, feature_extractor)
val_dataset = ImageDataset(val_images, val_prices, feature_extractor)

# function to override the default behavior of using torch.stack when making dataset, 
# because we are using different shapes for each sample
def custom_collate_fn(batch):
    pixel_values_list, prices = zip(*batch)
    return list(pixel_values_list), torch.tensor(prices, dtype=torch.float32)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=custom_collate_fn)


optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
epochs = EPOCHS
for epoch in range(epochs):
    train_loss = train(model, train_loader, optimizer, device, epoch)
    val_loss = validate(model, val_loader, device)
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# Save the model
torch.save(model.state_dict(), "models/vit_regression_model.pth")

# After training completion, for visualizing predictions
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
