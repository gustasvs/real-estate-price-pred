import os

def load_prices(count: int):
    """
    Load prices from a metadata file.

    Returns:
        A list of prices corresponding to each row in the file.
    """
    price_file = "data/metadata.txt"
    prices = []
    
    if os.path.exists(price_file):
        with open(price_file, 'r') as file:
            for line, i in zip(file, range(count)):
                parts = line.strip().split()
                if parts:
                    # The last element is the price
                    price = int(parts[-1])
                    prices.append(price)
    return prices

if __name__ == "__main__":
    prices = load_prices()
    print(prices)
