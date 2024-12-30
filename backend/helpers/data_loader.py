import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

from config.settings import BATCH_SIZE, ENABLE_DEV_LOGS, MAX_IMAGES_PER_SAMPLE, USE_ADDITIONAL_METADATA


class ImageDataset(Dataset):
    def __init__(self, inputs, prices, feature_extractor):
        self.inputs = inputs
        self.prices = prices
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        #* INPUT

        sample_images = self.inputs[idx][0] if USE_ADDITIONAL_METADATA else self.inputs[idx]

        # limit max images to 10 and select random choice of images
        num_images_to_select = min(len(sample_images), MAX_IMAGES_PER_SAMPLE)
        random_indices = np.random.choice(len(sample_images), num_images_to_select, replace=False)
        selected_images = [sample_images[idx] for idx in random_indices]
        # Preprocess and stack images
        sample_images_extracted = process_sample_images(selected_images, self.feature_extractor)
        if ENABLE_DEV_LOGS: print("current sample shape: ", sample_images_extracted.shape)

        # additional_metadata = [1, 1, 1, 1, 1]
        additional_metadata = self.inputs[idx][1] if USE_ADDITIONAL_METADATA else None

        sample_inputs = [sample_images_extracted, additional_metadata] if USE_ADDITIONAL_METADATA else sample_images_extracted

        #* TARGET
        price = self.prices[idx]

        return sample_inputs, torch.tensor(price, dtype=torch.float32)



def get_data_loaders(inputs, prices, feature_extractor):
    
    train_inputs, val_inputs, train_prices, val_prices = train_test_split(inputs, prices, test_size=0.1, random_state=42)
    train_dataset = ImageDataset(train_inputs, train_prices, feature_extractor)
    val_dataset = ImageDataset(val_inputs, val_prices, feature_extractor)

    # function to override the default behavior of using torch.stack when making dataset, 
    # because we are using different shapes for each sample
    def custom_collate_fn(batch):
        pixel_values_list, prices = zip(*batch)
        return list(pixel_values_list), torch.tensor(prices, dtype=torch.float32)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=custom_collate_fn)

    return train_loader, val_loader

def process_sample_images(images, feature_extractor):
    sample_images_extracted = torch.stack([
        feature_extractor(images=img, return_tensors="pt")['pixel_values'].squeeze(0) for img in images
    ])
    return sample_images_extracted
