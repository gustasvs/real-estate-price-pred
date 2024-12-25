import csv
import json
import os
from PIL import Image
import numpy as np

from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

# AVERAGE_DAYS_IN_MONTH = 30.44
AVERAGE_DAYS_IN_MONTH = 12

keywords = ["Cena", "Platība", "Istabas", "Stāvs", "Iela", "Rajons"]
csv_columns = keywords + ['URL', 'Images']

day_prices = []
month_prices = []

def extract_price(row, use_square_meters=False):
    price = row['Cena']

    price_str = row['Cena'].split(' ')[0]

    if "€" not in row['Cena'].split(' ')[1]:
        price_str += row['Cena'].split(' ')[1]

        if "€" not in row['Cena'].split(' ')[2]:
            price_str += row['Cena'].split(' ')[2]

    # Option to use price per square meter if requested
    if use_square_meters:
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

    # if price < 5 or price > 55:
    #     print(row)
    #     sample_images = extract_images(row, root_dir)
    #     num_images = len(sample_images)
    #     # all this to limit to 3 rows max
    #     num_rows = (num_images // 3) + (1 if num_images % 3 != 0 else 0)
    #     fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows))
    #     axes = axes.flatten() if num_rows > 1 else [axes]
    #     for ax, img in zip(axes, sample_images):
    #         if img is None:
    #             continue
            
    #         ax.imshow(img)
    #         # ax.set_title(f"Price: {price}")
    #         ax.axis('off')
    #     for ax in axes[len(sample_images):]:
    #         ax.axis('off')
    #     plt.title(f"Price: {price}")
    #     plt.show()

    # if float(price) > 40:
    #     print(row)

    return price

def extract_images(row, root_dir):
    image_list = json.loads(row['Images'])
    sample_images = []
    for image in image_list:
        image_path = f"{root_dir}images/{image}"
        if os.path.exists(image_path):
            image = Image.open(image_path).convert("RGB")

            if image is not None:
                sample_images.append(image)

    return sample_images

def extract_additional_metadata(row):
    additional_metadata = []

    try:

        # "Platība", "Istabas", "Stāvs", "Iela", "Rajons"
        square_meters = row['Platība'].split(' ')[0].strip()
        rooms_count = row['Istabas'].strip()
        aparment_floor = row['Stāvs'].split('/')[0].strip()
        building_floors = row['Stāvs'].split('/')[1].strip()
        has_elevator = 1 if "lifts" in row['Stāvs'] else 0
        street = row['Iela']
        district = row['Rajons']

        # if int(aparment_floor) > int(building_floors):
        #     print(f"Apartment floor is higher than building floors: {row['URL'].split('/')[-1]} {aparment_floor} > {building_floors} ")
        #     print("Swapping the values")
        #     temp_apartment_floor = aparment_floor
        #     aparment_floor = building_floors
        #     building_floors = temp_apartment_floor

        additional_metadata = [square_meters, rooms_count, aparment_floor, building_floors, has_elevator, street, district]

    except Exception as e:
        print(f"Error: {e}")
        print(row)

    return additional_metadata

def normalise_and_prepare_additional_metadata(additional_metadata):
    # "Platība", "Istabas", "Stāvs", "Iela", "Rajons"
    
    square_meters = [float(metadata[0]) for metadata in additional_metadata]
    rooms_count = [int(metadata[1]) for metadata in additional_metadata]
    apartment_floor = [int(metadata[2]) for metadata in additional_metadata]
    building_floors = [int(metadata[3]) for metadata in additional_metadata]
    has_elevator = [metadata[4] for metadata in additional_metadata]
    street = [metadata[5] for metadata in additional_metadata] # Not used for now
    district = [metadata[6] for metadata in additional_metadata]

    print(f"Square Meters: {np.array(square_meters).shape} \n Rooms Count: {np.array(rooms_count).shape} \n Apartment Floor: {np.array(apartment_floor).shape} \n Building Floors: {np.array(building_floors).shape} \n Has Elevator: {np.array(has_elevator).shape} \n Street: {np.array(street).shape} \n District: {np.array(district).shape}")
    floor_ratio = []
    for ap_floor, bldg_floors in zip(apartment_floor, building_floors):
        if bldg_floors == 0:
            floor_ratio.append(0)
        else:
            floor_ratio.append(ap_floor / bldg_floors)

    def min_max_scale(data):
        # return (np.array(data) - min(data)) / (max(data) - min(data))
        return (np.array(data) - np.mean(data)) / np.std(data)
        # return np.array(data)

    square_meters = min_max_scale(square_meters)
    rooms_count = min_max_scale(rooms_count)
    apartment_floor = min_max_scale(apartment_floor)
    building_floors = min_max_scale(building_floors)


    unique_districts = sorted(set(district))
    print(f"Unique Districts: {unique_districts}")
    encoder = OneHotEncoder(categories=[unique_districts], sparse_output=False)
    # district_one_hot = encoder.fit_transform(np.array(district).reshape(-1, 1))
    # district_one_hot = np.array([1 for _ in range(len(district))])
    district_one_hot = np.array([unique_districts.index(d) for d in district])

    district_one_hot = min_max_scale(district_one_hot)

    print(f"District One Hot Shape: {district_one_hot.shape}")
    print(f"Rooms Count Shape: {np.array(rooms_count).shape}")
    print(f"Apartment Floor Shape: {np.array(apartment_floor).shape}")
    print(f"Building Floors Shape: {np.array(building_floors).shape}")
    print(f"Floor Ratio Shape: {np.array(floor_ratio).shape}")
    print(f"Has Elevator Shape: {np.array(has_elevator).shape}")

    output_metadata = np.column_stack((square_meters, rooms_count, apartment_floor, building_floors, floor_ratio, has_elevator, district_one_hot))

    # plt.hist(square_meters, bins=20)
    # plt.title("Square Meters distribution")
    # plt.show()

    # plt.hist(rooms_count, bins=20)
    # plt.title("Rooms Count distribution")
    # plt.show()  

    # fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    # ax[0].hist(apartment_floor, bins=20)
    # ax[0].set_title("Apartment Floor distribution")
    # ax[1].hist(building_floors, bins=20)
    # ax[1].set_title("Building Floors distribution")
    # plt.show()

    # plt.hist(floor_ratio, bins=20)
    # plt.title("Floor Ratio distribution")
    # plt.show()

    # plt.hist(has_elevator, bins=20)
    # plt.title("Has Elevator distribution")
    # plt.show()

    return output_metadata


def extract_images_and_prices(count: int, root_dir: str, use_square_meters: bool):

    prices = []
    images = []
    additional_metadata = []

    with open(f'{root_dir}real_estate_data.csv', 'r') as csv_file:
        reader = csv.DictReader(csv_file, fieldnames=csv_columns)        
        
        for index, row in enumerate(reader):
            if index >= count:
                break

            if "Citi" in row['Istabas']:
                continue
            
            row_price = extract_price(row, use_square_meters)
            row_images = extract_images(row, root_dir)
            row_additional_metadata = extract_additional_metadata(row)
        
            prices.append(float(row_price))
            images.append(row_images)
            additional_metadata.append(row_additional_metadata)

    additional_metadata = normalise_and_prepare_additional_metadata(additional_metadata)
    

    # mean_day_prices = np.mean(day_prices)
    # mean_month_prices = np.mean(month_prices)
    # print(f"Mean Day Price: {mean_day_prices}")
    # print(f"Mean Month Price: {mean_month_prices}")
    # plt.figure(figsize=(10, 5))
    # plt.hist(day_prices, bins=20, alpha=0.5, label='Day Prices')
    # plt.hist(month_prices, bins=20, alpha=0.5, label='Month Prices')
    # plt.axvline(mean_day_prices, color='blue', linestyle='dashed', linewidth=2, label=f'Mean Day Price: {mean_day_prices:.2f}')
    # plt.axvline(mean_month_prices, color='red', linestyle='dashed', linewidth=2, label=f'Mean Month Price: {mean_month_prices:.2f}')
    # plt.legend(loc='upper right')
    # plt.title("Price Distribution: Day vs. Month")
    # plt.xlabel("Price")
    # plt.ylabel("Frequency")
    # plt.show()



    # plt.hist(prices, bins=40)
    # plt.title("Price distribution before removing outliers")
    # mean_price = np.mean(prices)
    # std_price = np.std(prices)
    # for i in range(1, 4):
    #     plt.axvline(mean_price + i * std_price, color='r', linestyle='dashed', linewidth=1)
    #     plt.axvline(mean_price - i * std_price, color='r', linestyle='dashed', linewidth=1)
    #     plt.text(mean_price + i * std_price, plt.ylim()[1] * 0.9, f'+{i}σ', color='r')
    #     plt.text(mean_price - i * std_price, plt.ylim()[1] * 0.9, f'-{i}σ', color='r')

    # plt.show()
    
    return prices, images, additional_metadata


if __name__ == "__main__":
    prices, images = extract_images_and_prices(666, root_dir = "", use_square_meters = True)
