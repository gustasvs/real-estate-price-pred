import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

from config.settings import BATCH_SIZE, ENABLE_DEV_LOGS, LEARNING_RATE, LOSS_FUNCT, EPOCHS, BATCHES_TO_AGGREGATE


class ImageDataset(Dataset):
    def __init__(self, images, prices, feature_extractor):
        self.images = images 
        self.prices = prices
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        sample_images = self.images[idx]

        # sampled_count = np.random.randint(2, len(sample_images) + 1)
        # sample_images = np.random.choice(sample_images, sampled_count, replace=False)

        price = self.prices[idx]
        
        # Preprocess and stack images
        sample_images_extracted = torch.stack([
            self.feature_extractor(images=img, return_tensors="pt")['pixel_values'].squeeze(0)
            for img in sample_images
        ])

        if ENABLE_DEV_LOGS: print("current sample shape: ", sample_images_extracted.shape)

        return sample_images_extracted, torch.tensor(price, dtype=torch.float32)



def get_data_loaders(images, prices, feature_extractor):
    
    
    train_images, val_images, train_prices, val_prices = train_test_split(images, prices, test_size=0.1)
    train_dataset = ImageDataset(train_images, train_prices, feature_extractor)
    val_dataset = ImageDataset(val_images, val_prices, feature_extractor)

    # function to override the default behavior of using torch.stack when making dataset, 
    # because we are using different shapes for each sample
    def custom_collate_fn(batch):
        pixel_values_list, prices = zip(*batch)
        return list(pixel_values_list), torch.tensor(prices, dtype=torch.float32)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=custom_collate_fn)

    return train_loader, val_loader