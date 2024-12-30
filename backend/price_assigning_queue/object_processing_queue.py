import amqp
import json

from PIL import Image
import requests
from io import BytesIO

import matplotlib.pyplot as plt

from helpers.processed_data import scale_metadata_for_sample, descale_price

from price_assigning_queue.compute_predicted_price import compute_predicted_price

# so config can be imported
import sys
from pathlib import Path
current_file_path = Path(__file__).absolute()
project_root = current_file_path.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))


from price_assigning_queue.handle_db_operations import get_object_images_from_db, get_all_residences, update_object_predicted_price_in_db, get_object_metadata_from_db
from price_assigning_queue.handle_minio_storage_operations import create_minio_client, generate_presigned_url, fetch_images_from_presigned_urls

def object_processing_queue():

    get_all_residences()

    minio_client = create_minio_client()

    with amqp.Connection(host='localhost') as c:
        ch = c.channel()
        def on_message(message):


            print('Received message (delivery tag: {}): {}'.format(message.delivery_tag, message.body))

            # STEP 1: extract object id from message
            message_body = json.loads(message.body)
            object_id = message_body.get('objectId')


            # STEP 2.1: get image urls from postgres using object id
            residence_image_urls = get_object_images_from_db(object_id)

            if not residence_image_urls or len(residence_image_urls) == 0:
                print("No images found for object with id: ", object_id)
                ch.basic_ack(message.delivery_tag)
                return
            print(f"Fetched {len(residence_image_urls)} images for object with id: {object_id}")

            residence_image_pre_signed_urls = []
            for image_url in residence_image_urls:
                # print("Image URL: ", image_url)
                presigned_url = generate_presigned_url(minio_client, image_url, 'object-pictures')
                # print("Pre-signed URL: ", presigned_url)
                residence_image_pre_signed_urls.append(presigned_url)
            images = fetch_images_from_presigned_urls(residence_image_pre_signed_urls)
            # fig = plt.figure(figsize=(10, 10))
            # rows = len(images) // 2
            # columns = 2
            # for i, image in enumerate(images):
            #     fig.add_subplot(rows, columns, i + 1)
            #     plt.imshow(image)
            # plt.show()

            # STEP 2.2: get metadata from postgres using object id
            metadata = get_object_metadata_from_db(object_id)

            if not metadata:
                print("No metadata found for object with id: ", object_id)
                ch.basic_ack(message.delivery_tag)
                return

            print("Metadata: ", metadata)
            print("-" * 20)

            scaled_metadata_for_sample = scale_metadata_for_sample(metadata)

            print("Metadata: ", metadata)
            print("Descaled Metadata: ", scaled_metadata_for_sample)

            # STEP 4: convert images to a price estimate using the VIT model
            prediction = compute_predicted_price(images, scaled_metadata_for_sample)
            print("Prediction: ", prediction)

            descaled_price = descale_price(prediction)
            print("Descaled price: ", descaled_price)
            # predicted_price = get_price_estimate_from_vit_model(images)
            predicted_price = descaled_price * metadata['area']
            
            # STEP 5: update postgres databases object with calculated price prediction
            update_object_predicted_price_in_db(object_id, predicted_price)

            # STEP 6: confirm message and wait for next message
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