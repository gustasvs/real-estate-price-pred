import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam
from transformers import ViTImageProcessor, ViTModel
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import torchvision.transforms as T
from sklearn.preprocessing import MinMaxScaler

import time

import transformers
transformers.logging.set_verbosity_error()



from load_images import load_images
from load_prices import load_prices


# Define the custom dataset
class ImageDataset(Dataset):
    def __init__(self, images, prices, feature_extractor):
        self.images = images  # List of image paths or PIL images
        self.prices = prices  # List of prices
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        sample_images = self.images[idx]  # List of images for the current instance
        price = self.prices[idx]
        
        # Preprocess and stack images
        sample_images_extracted = torch.stack([
            self.feature_extractor(images=img, return_tensors="pt")['pixel_values'].squeeze(0)
            for img in sample_images
        ])

        print("Current instance pixel values shape: ", sample_images_extracted.shape)

        return sample_images_extracted, torch.tensor(price, dtype=torch.float32)


# Load the base ViT model without head
base_model = ViTModel.from_pretrained("google/vit-base-patch16-224")



class ImageAggregator(nn.Module):
    def __init__(self, aggregation_method="mean"):
        super(ImageAggregator, self).__init__()
        self.aggregation_method = aggregation_method

    def forward(self, embeddings):
        if self.aggregation_method == "mean":
            return torch.mean(embeddings, dim=0);
            # return torch.mean(embeddings, dim=1)  # Correct dimension for mean pooling
        # Implement other methods as needed
        return embeddings  # Fallback



class CustomViTHead(nn.Module):
    def __init__(self):
        super(CustomViTHead, self).__init__()
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, 1)  # Ensure output shape is [batch_size, 1]

    def forward(self, x):
        x = self.dropout(x)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x.squeeze(-1)  # Change here to ensure output shape matches target shape




# Combine base model with custom head
class ViTMultiImageRegressionModel(nn.Module):
    def __init__(self, base_model, aggregator, custom_head):
        super(ViTMultiImageRegressionModel, self).__init__()
        self.base_model = base_model
        self.aggregator = aggregator
        self.custom_head = custom_head

    def forward(self, batch):
        batch_embeddings = []
        
        # create embeddings for all images in one sample in batch
        for sample in batch:
            # You only need to stack if instance_images is a list of tensors, otherwise directly use the tensor.
            instance_embeddings = [self.base_model(pixel_values=image.unsqueeze(0)).last_hidden_state[:, 0, :]
                                   for image in sample]
            instance_embeddings = torch.cat(instance_embeddings, dim=0)
            batch_embeddings.append(instance_embeddings)
        
        aggregated_embeddings = torch.stack([
            self.aggregator(instance) for instance in batch_embeddings
        ])

        print("Aggregated embeddings shape: ", aggregated_embeddings.shape)
        
        # Pass aggregated embedding through regression head
        output = self.custom_head(aggregated_embeddings)
        print("Final output shape: ", output.shape)
        return output




aggregator = ImageAggregator(aggregation_method="mean")
custom_head = CustomViTHead()
model = ViTMultiImageRegressionModel(base_model, aggregator, custom_head)

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


# Modify the train function to include this display
def train(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0
    start_time = time.time()
    total_batches = len(dataloader)
    
    for batch_idx, (sample, prices) in enumerate(dataloader, 1):

        sample = [instance.to(device) for instance in sample]

        print("one input in batch shape: ", sample[0].shape)  # Should print [8, 2, 3, 224, 224] for a batch size of 8

        prices = prices.to(device)

        print("prices shape (should be batch size): ", prices.shape)

        optimizer.zero_grad()

        outputs = model(sample)
        loss = criterion(outputs, prices) 
        loss.backward()
        optimizer.step()

        # Update running loss
        running_loss += loss.item()

        # Display the first image in the batch once per epoch (optional)
        # if batch_idx == 1:
        #     show_image_from_tensor(pixel_values[0])

        # Log detailed information per batch
        if batch_idx % 10 == 0 or batch_idx == total_batches:
            elapsed_time = time.time() - start_time
            avg_loss = running_loss / batch_idx
            print(
                f"Epoch [{epoch+1}] - Batch [{batch_idx}/{total_batches}]: "
                f"Batch Loss = {loss.item():.4f}, Avg Loss = {avg_loss:.4f}, "
                f"Elapsed Time = {elapsed_time:.2f}s"
            )

    return running_loss / total_batches


# Validation setup
def validate(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for pixel_values_list, prices in dataloader:
            pixel_values_list = [instance.to(device) for instance in pixel_values_list]

            prices = prices.to(device)
            
            outputs = model(pixel_values_list)
            loss = criterion(outputs.squeeze(), prices)
            val_loss += loss.item()
    return val_loss / len(dataloader)

# Main script
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)

# Example data and preprocessing
# Replace with actual image data and prices

count = 100

images = load_images(count)
# shape = [count, image_count, image]


# prices = [np.random.randint(100, 1000) for _ in range(count)]
prices = load_prices(count)

# scaler = MinMaxScaler()
# prices = scaler.fit_transform(np.array(prices).reshape(-1, 1)).flatten()
prices = np.array(prices)


feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

train_images, val_images, train_prices, val_prices = train_test_split(images, prices, test_size=0.2)

train_dataset = ImageDataset(train_images, train_prices, feature_extractor)
val_dataset = ImageDataset(val_images, val_prices, feature_extractor)

# function to override the default behavior of using torch.stack, because we are using different image counts
def custom_collate_fn(batch):
    pixel_values_list, prices = zip(*batch)
    return list(pixel_values_list), torch.tensor(prices, dtype=torch.float32)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=custom_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=4, collate_fn=custom_collate_fn)


criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=1e-4)

# Training loop
epochs = 10
for epoch in range(epochs):
    train_loss = train(model, train_loader, criterion, optimizer, device, epoch)
    val_loss = validate(model, val_loader, criterion, device)
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# Save the model
torch.save(model.state_dict(), "models/vit_regression_model.pth")
