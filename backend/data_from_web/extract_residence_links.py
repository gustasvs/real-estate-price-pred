import requests
from bs4 import BeautifulSoup


MAX_PAGES = 47

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0',
    'Accept': 'image/avif,image/webp,image/png,image/svg+xml,image/*;q=0.8,*/*;q=0.5',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate, br, zstd',
    'Referer': 'https://www.city24.lv/',
    'Sec-Fetch-Dest': 'image',
    'Sec-Fetch-Mode': 'no-cors',
    'Sec-Fetch-Site': 'cross-site',
    'Connection': 'keep-alive',
    'TE': 'trailers'
}

def fetch_item_links(url):

    original_url = url
    links = []

    current_page = 1

    while url:
        print("Fetching:", url)
        response = requests.get(url, headers=HEADERS)

        if response.status_code != 200:
            print(f"Failed to fetch {url}. Status code: {response.status_code}")
            break
        
        soup = BeautifulSoup(response.text, 'html.parser')

        # find item containers
        for item_container in soup.find_all('div', class_='d1'):
            # find all <a> elements in item container            
            for link in item_container.find_all('a'):
                item_url = link.get('href')
                # print(item_url)
                if item_url:
                    full_url = f"https://www.ss.lv/{item_url}"
                    links.append(full_url)
        
        current_page += 1
        if current_page > MAX_PAGES:
            break
        
        # first page is not numbered and starting from the second page the numbering starts from 2
        # url = f"https://www.ss.lv/lv/real-estate/flats/riga/centre/hand_over/page{current_page}.html"
        url = f"{original_url}page{current_page}.html"

    return links

def save_links_to_file(links, filename='links.txt'):
    existing_links = set()
    try:
        with open(filename, 'r') as file:
            existing_links = {line.strip() for line in file if line.strip()}
    except FileNotFoundError:
        print("No existing file found. A new one will be created.")

    new_links = [link for link in links if link not in existing_links]

    updated_links = existing_links.union(set(links))
    # Write new links to the file
    with open(filename, 'w') as file:
        for link in updated_links:
            file.write(link + '\n')

    print(f"Added {len(updated_links) - len(existing_links) } new links to the file, Total unique links: {len(updated_links)}")



if __name__ == '__main__':
    # start_url = 'https://www.ss.lv/lv/real-estate/flats/riga/centre/hand_over/'
    # start_url = 'https://www.ss.lv/lv/real-estate/flats/riga/agenskalns/hand_over/'
    # start_url = 'https://www.ss.lv/lv/real-estate/flats/riga/teika/hand_over/'
    # start_url = 'https://www.ss.lv/lv/real-estate/flats/riga/yugla/hand_over/'
    # start_url = 'https://www.ss.lv/lv/real-estate/flats/riga/ziepniekkalns/hand_over/'
    # start_url = 'https://www.ss.lv/lv/real-estate/flats/riga/purvciems/hand_over/'
    # start_url = 'https://www.ss.lv/lv/real-estate/flats/riga/mezhapark/hand_over/'
    # start_url = 'https://www.ss.lv/lv/real-estate/flats/riga/mezhciems/hand_over/'
    start_url = 'https://www.ss.lv/lv/real-estate/flats/riga/all/hand_over/'
    all_links = fetch_item_links(start_url)
    save_links_to_file(all_links)

