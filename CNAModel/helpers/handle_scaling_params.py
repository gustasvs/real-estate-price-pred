import json

def handle_scaling_params(name, parameters=None, save=False, file_path='scaling-params.json'):
    # Check if the file exists to avoid errors on reading non-existent file
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {}

    if save:
        data[name] = parameters
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
    else:
        return data.get(name, None)
