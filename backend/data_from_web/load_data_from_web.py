import csv
import json
import os
from PIL import Image
import numpy as np

import matplotlib.pyplot as plt

# AVERAGE_DAYS_IN_MONTH = 30.44
AVERAGE_DAYS_IN_MONTH = 12
USE_SQUARE_METERS = True
# USE_SQUARE_METERS = False

keywords = ["Cena", "Platība", "Istabas", "Stāvs", "Iela", "Rajons"]
csv_columns = keywords + ['URL', 'Images']

day_prices = []
month_prices = []

def extract_images_and_prices(count: int, root_dir: str):

    prices = []
    images = []

    with open(f'{root_dir}real_estate_data.csv', 'r') as csv_file:
        reader = csv.DictReader(csv_file, fieldnames=csv_columns)        
        
        for index, row in enumerate(reader):
            if index >= count:
                break

            price = row['Cena']
            image_list = json.loads(row['Images'])

            price_str = row['Cena'].split(' ')[0]

            if "€" not in row['Cena'].split(' ')[1]:
                price_str += row['Cena'].split(' ')[1]

                if "€" not in row['Cena'].split(' ')[2]:
                    price_str += row['Cena'].split(' ')[2]

            # Option to use price per square meter if requested
            if USE_SQUARE_METERS:
                price_per_sqm = float(row['Cena'].split('(')[1].split(' ')[0])
                price_str = price_per_sqm

            price_type = "day" if "dienā" in row['Cena'] else "month"

            # for now skip the rows with price per day
            # if price_type == "day":
            #     continue

            if price_type == "day":
                price = float(price_str) * AVERAGE_DAYS_IN_MONTH
                day_prices.append(price)
            else:
                price = float(price_str)
                month_prices.append(price)

            sample_images = []
            for image in image_list:
                image_path = f"{root_dir}images/{image}"
                if os.path.exists(image_path):
                    image = Image.open(image_path).convert("RGB")
                    # image = image.resize((224, 224))
                    sample_images.append(image)

            # if price > 50:
            #     print(row)
            #     num_images = len(sample_images)
            #     num_rows = (num_images // 3) + (1 if num_images % 3 != 0 else 0)
            #     fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows))
            #     axes = axes.flatten() if num_rows > 1 else [axes]
            #     for ax, img in zip(axes, sample_images):
            #         ax.imshow(img)
            #         # ax.set_title(f"Price: {price}")
            #         ax.axis('off')
            #     for ax in axes[len(sample_images):]:
            #         ax.axis('off')
            #     plt.title(f"Price: {price}")
            #     plt.show()
            
            prices.append(float(price))

            # limit images to max 5
            # if len(sample_images) > 10:
            #     sample_images = sample_images[:10]

            images.append(sample_images)


    
    mean_day_prices = np.mean(day_prices)
    mean_month_prices = np.mean(month_prices)
    print(f"Mean Day Price: {mean_day_prices}")
    print(f"Mean Month Price: {mean_month_prices}")

    plt.figure(figsize=(10, 5))
    plt.hist(day_prices, bins=20, alpha=0.5, label='Day Prices')
    plt.hist(month_prices, bins=20, alpha=0.5, label='Month Prices')

    plt.axvline(mean_day_prices, color='blue', linestyle='dashed', linewidth=2, label=f'Mean Day Price: {mean_day_prices:.2f}')
    plt.axvline(mean_month_prices, color='red', linestyle='dashed', linewidth=2, label=f'Mean Month Price: {mean_month_prices:.2f}')

    plt.legend(loc='upper right')
    plt.title("Price Distribution: Day vs. Month")
    plt.xlabel("Price")
    plt.ylabel("Frequency")
    plt.show()



    plt.hist(prices, bins=40)
    plt.title("Price distribution before removing outliers")
    mean_price = np.mean(prices)
    std_price = np.std(prices)
    for i in range(1, 4):
        plt.axvline(mean_price + i * std_price, color='r', linestyle='dashed', linewidth=1)
        plt.axvline(mean_price - i * std_price, color='r', linestyle='dashed', linewidth=1)
        plt.text(mean_price + i * std_price, plt.ylim()[1] * 0.9, f'+{i}σ', color='r')
        plt.text(mean_price - i * std_price, plt.ylim()[1] * 0.9, f'-{i}σ', color='r')

    plt.show()
    
    return prices, images


if __name__ == "__main__":
    prices, images = extract_images_and_prices(666, root_dir="")
