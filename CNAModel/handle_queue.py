from dotenv import load_dotenv
import os

# Load variables from .env file
load_dotenv()


from price_assigning_queue.object_processing_queue import object_processing_queue

object_processing_queue()