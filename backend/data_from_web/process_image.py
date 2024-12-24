from PIL import Image
import io

def resize_to_square(image, target_size=244):
    """
    This will first resize to target_size the smallest side of the image 
    and then crop longest side to target_size from both sides equally.
    """
    scaling_factor = target_size / min(image.width, image.height)
    
    new_width = int(image.width * scaling_factor)
    new_height = int(image.height * scaling_factor)
    
    image = image.resize((new_width, new_height), Image.ANTIALIAS)
    
    left = (new_width - target_size) / 2
    top = (new_height - target_size) / 2
    right = (new_width + target_size) / 2
    bottom = (new_height + target_size) / 2
    
    image = image.crop((left, top, right, bottom))
    return image

def process_image(image):
    image = Image.open(io.BytesIO(image)).convert("RGB")
    image = resize_to_square(image)
    return image