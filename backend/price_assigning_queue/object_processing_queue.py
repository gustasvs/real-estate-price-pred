import amqp
import json

from PIL import Image
import requests
from io import BytesIO

import matplotlib.pyplot as plt


# so config can be imported
import sys
from pathlib import Path
current_file_path = Path(__file__).absolute()
project_root = current_file_path.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))


from price_assigning_queue.handle_db_operations import get_object_images_from_db, get_all_residences
from price_assigning_queue.handle_minio_storage_operations import create_minio_client, generate_presigned_url, fetch_images_from_presigned_urls

def object_processing_queue():

    # get_all_residences()

    minio_client = create_minio_client()

    with amqp.Connection(host='localhost') as c:
        ch = c.channel()
        def on_message(message):
            print('Received message (delivery tag: {}): {}'.format(message.delivery_tag, message.body))

            message_body = json.loads(message.body)
            object_id = message_body.get('objectId')
            print("Object ID: ", object_id)

            residence_image_urls = get_object_images_from_db(object_id)

            residence_image_pre_signed_urls = []

            for image_url in residence_image_urls:
                print("Image URL: ", image_url)
                # Generate the pre-signed URL for the image
                presigned_url = generate_presigned_url(minio_client, image_url, 'object-pictures')
                print("Pre-signed URL: ", presigned_url)
                residence_image_pre_signed_urls.append(presigned_url)

            # image = Image.open(image_path).convert("RGB")
            #     image = image.resize((224, 224))
            #     sample_images.append(image)
            images = fetch_images_from_presigned_urls(residence_image_pre_signed_urls)

            # plt show
            fig = plt.figure(figsize=(10, 10))
            rows = len(images) // 2
            columns = 2
            for i, image in enumerate(images):
                fig.add_subplot(rows, columns, i + 1)
                plt.imshow(image)

            plt.show()


            ch.basic_ack(message.delivery_tag)
        ch.queue_declare(queue='objectCreationQueue', durable=True, auto_delete=False)
        ch.basic_consume(queue='objectCreationQueue', callback=on_message)
        print('Waiting for messages. To exit press CTRL+C')
        try:
            while True:
                c.drain_events()
        except KeyboardInterrupt:
            print('Exiting...')

        # images = get_object_images_from_db(object_id)
        # print(images)