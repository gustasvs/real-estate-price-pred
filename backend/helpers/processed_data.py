import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from load_images import load_images
from load_prices import load_prices

from helpers.label_smothing import apply_lds, apply_fds, adaptive_lds

from data_from_web.load_data_from_web import extract_images_and_prices

from helpers.handle_scaling_params import handle_scaling_params

from config.settings import DEMO_MODE, STANDART_DEV_TO_KEEP, USE_SQUARE_METERS, USE_ADDITIONAL_METADATA

def processed_data(count):
    """
    returns inputs and targets scaled, normalized 
    """
    
    # CALIFORNIA HOUSING DATASET
    # images = load_images(count) # shape = [count, image_count, image]
    # print("Images loaded...")
    # prices = load_prices(count) # shape = [count]
    # print("Prices loaded...")

    # LATVIAN REAL ESTATE DATASET
    prices, images, additional_metadata = extract_images_and_prices(count, root_dir="data_from_web/", use_square_meters=USE_SQUARE_METERS)

    print(f"Prices count: {len(prices)}, Images count: {len(images)}, Additional metadata count: {len(additional_metadata)}")

    # filter out outliers
    mean_price = np.mean(prices)
    std_price = np.std(prices)
    print(f"Mean price: {mean_price}, std price: {std_price}")
    print(f"thresholds: {mean_price - STANDART_DEV_TO_KEEP * std_price} and {mean_price + STANDART_DEV_TO_KEEP * std_price}")
    
    
    filtered_indices = [i for i in range(len(prices)) if mean_price - STANDART_DEV_TO_KEEP * std_price < prices[i] < mean_price + STANDART_DEV_TO_KEEP * std_price]
    filtered_prices = [prices[i] for i in filtered_indices]
    filtered_images = [images[i] for i in filtered_indices]
    filtered_additional_metadata = [additional_metadata[i] for i in filtered_indices]

    prices = filtered_prices
    images = filtered_images
    additional_metadata = filtered_additional_metadata

    count = len(prices)

    print(f"*" * 20)
    print(f"Filtered count: {count}")
    print(f"*" * 20)

    # if DEMO_MODE:
    #     plt.hist(prices, bins=20)
    #     plt.title("Price distribution after removing outliers")
    #     plt.show()

    prices = np.array(prices)

    # prices = apply_fds(prices, prices, sigma=1.4)

    # if DEMO_MODE:
    #     while True:
    #         sigma = float(input("Enter the sigma value for LDS (0 to stop): "))
    #         if sigma == 0: break

    #         # density_threshold = float(input("Enter the density threshold for adaptive LDS: "))
    #         # max_sigma = float(input("Enter the max sigma value for adaptive LDS: "))
    #         # min_sigma = float(input("Enter the min sigma value for adaptive LDS: "))

    #         fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    #         fig.canvas.manager.window.wm_geometry("+10+10")
            
    #         ax[0].hist(prices, bins=20)
    #         ax[0].set_title("Original Prices")
    #         # ax[1].hist(apply_lds(prices, sigma=sigma), bins=50)
    #         # ax[1].hist(adaptive_lds(prices, density_threshold=density_threshold, max_sigma=max_sigma, min_sigma=min_sigma), bins=50)
    #         smoothed_prices = apply_fds(prices, prices, sigma=sigma)
    #         ax[1].hist(smoothed_prices, bins=20)
    #         ax[1].set_title("Prices after fds (Applying Feature Distribution Smoothing (FDS) by aligning features to smoothed labels)")
    #         ax[2].hist(apply_lds(prices, sigma=sigma), bins=20)
    #         ax[2].set_title("Plain Lds (Applying Label Distribution Smoothing (LDS) on the results array)")

    #         fig.subplots_adjust(hspace=0.5)
    #         plt.show()

    scaler = MinMaxScaler()
    prices = scaler.fit_transform(np.array(prices).reshape(-1, 1)).flatten()

    # save the scaling parameters
    handle_scaling_params("prices", {"min": scaler.data_min_[0], "max": scaler.data_max_[0]}, save=True)

    # redefine stuff function returns clearer
    inputs = np.array([[images[i], additional_metadata[i]] for i in range(len(images))]) if USE_ADDITIONAL_METADATA else images
    targets = prices

    return inputs, targets

