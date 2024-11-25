import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from load_images import load_images
from load_prices import load_prices

from helpers.label_smothing import apply_lds, apply_fds, adaptive_lds

def processed_data(count):
    """
    returns inputs and targets scaled, normalized 
    """
    
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

    prices = np.array(prices)

    prices = apply_fds(prices, prices, sigma=1.4)

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

    # plt.hist(prices, bins=10)
    # plt.show()

    scaler = MinMaxScaler()
    prices = scaler.fit_transform(np.array(prices).reshape(-1, 1)).flatten()


    return images, prices

