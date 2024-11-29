# so config can be imported
import sys
from pathlib import Path
current_file_path = Path(__file__).absolute()
project_root = current_file_path.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))


from price_assigning_queue.handle_db_operations import get_object_images_from_db, get_all_residences

def object_processing_queue():

    get_all_residences()

    queue = ["bb62b67c-228e-4ed5-9bb4-9e64d26cf070"]
    for object_id in queue:
        images = get_object_images_from_db(object_id)
        print(images)