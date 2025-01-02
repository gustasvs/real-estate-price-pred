import os
import csv
import json
import requests
import time
from tqdm import tqdm  
from bs4 import BeautifulSoup

from extract_residence_links import HEADERS
from process_image import process_image


keywords = ["Cena", "Platība", "Istabas", "Stāvs", "Iela", "Rajons"]

csv_columns = keywords + ['URL', 'Images']


def extract_image_links_from_soup(soup):
    image_links = []
    for div in soup.find_all('div', class_='pic_dv_thumbnail'):
        a_tag = div.find('a')
        if a_tag and 'href' in a_tag.attrs:
            image_links.append(a_tag['href'])
    
    return image_links

def process_property_value(label, value):
    if label == "Iela":
        # usually has [Karte] at the end, and we want to remove it
        return value.split("[")[0].strip()
    
    # implement other processing rules here
    else:
        return value

def extract_property_metadata_from_soup(soup):
    property_metadata = {}
    
    for tr in soup.find_all('tr'):
        tds = tr.find_all('td')
        if len(tds) == 2:
            label = tds[0].text.strip()
            value = tds[1].text.strip()
            
            label_for_comparison = label.split(':')[0]
            
            if label_for_comparison in keywords:
                property_metadata[label_for_comparison] = process_property_value(label_for_comparison, value)
    
    return property_metadata

def analyze_real_estate(url):
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        # print(f"Failed to fetch {url}. Status code: {response.status_code}")
        return [], []
    
    soup = BeautifulSoup(response.text, 'html.parser')

    image_links = extract_image_links_from_soup(soup)
    
    image_metadata = extract_property_metadata_from_soup(soup)

    return image_metadata, image_links

def get_correct_format(value):
    format_map = {
        'jpg': 'JPEG',
        'jpeg': 'JPEG',
        'png': 'PNG',
        'gif': 'GIF'
    }
    return format_map.get(value, 'JPEG')

def download_image(property_url, image_url):
    # create images directory if it doesn't exist
    if not os.path.exists('images'):
        os.makedirs('images')
    
    response = requests.get(image_url, headers=HEADERS)

    if response.status_code != 200:
        print(f"Failed to download image. Status code: {response.status_code}")
        return None
    
    if not response.content:
        print(f"Failed to download image. No content.")
        return None

    finalised_image_path = f"{property_url.split('/')[-1].split('.')[0]}-{image_url.split('/')[-1]}"
    file_extension = image_url.split('.')[-1] if '.' in image_url.split('/')[-1] else 'jpg'



    processed_image = process_image(response.content)
    processed_image.save(f"images/{finalised_image_path}",
                         format=get_correct_format(file_extension))

    return finalised_image_path

def check_if_already_handled(url):

    url_in_csv = False
    url_images_processed = False

    with open('real_estate_data.csv', 'r') as csv_file:
        reader = csv.DictReader(csv_file, fieldnames=csv_columns)

        # print(reader.fieldnames)

        if 'URL' not in reader.fieldnames:
            return False, "URL column not found in CSV."
        for row in reader:
            if row['URL'] == url:
                url_in_csv = True
                break

    url_identifier = url.split('/')[-1].split('.')[0]

    if url_in_csv:
        image_files = os.listdir('images')
        url_images_processed = any(file.startswith(url_identifier) for file in image_files)

    return url_in_csv or url_images_processed, f"{url_in_csv and 'URL already in CSV.'} {url_images_processed and 'Images already processed.'}"
    
        

def process_links(links_file, csv_file):

    try:
        pbar = tqdm(links_file, desc="Processing links", total=940)
        for url in pbar:
            url = url.strip()
            
            if not url:
                continue

            already_handled, message = check_if_already_handled(url)
            if already_handled:
                # print(f"{url.split('/')[-1]}: {message}")
                # pbar.set_description(f"{url.split('/')[-1]}: {message}")
                continue
            
            # print("Processing:", url, end=': ')

            # continue

            property_metadata, image_links = analyze_real_estate(url)

            if not property_metadata:
                print(f"Failed to process {url}. Skipping.")
                continue

            data_to_write = {
                'URL': url,
                **property_metadata  # simple way to unpack the metadata
            }

            # download and store images
            storage_paths_for_images = []
            for image_link in image_links:
                image_path = download_image(url, image_link)
                
                if image_path:
                    storage_paths_for_images.append(image_path)

            data_to_write['Images'] = json.dumps(storage_paths_for_images)

            writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
            writer.writerow(data_to_write)

            # print(" image-count:", len(storage_paths_for_images), " time:", round(time.time() - start_time, 2), "s")

            pbar.set_description(f"{url.split('/')[-1]} Images: {len(storage_paths_for_images)}")


    except FileNotFoundError:
        print("File not found. Please check the filename and path.")
    except KeyboardInterrupt:
        print("Process interrupted.")
    except Exception as e:
        print("An error occurred:", e)

    finally:
        # cleanup
        links_file.close()
        csv_file.close()

        print("Finished processing.")



if __name__ == '__main__':

    links_file = open('links.txt', 'r')
    csv_file = open('real_estate_data.csv', 'a', newline='')

    process_links(links_file, csv_file)