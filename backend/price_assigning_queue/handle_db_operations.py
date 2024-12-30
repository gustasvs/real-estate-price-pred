import psycopg2
from psycopg2.extras import RealDictCursor
import os

# Database connection configuration
DATABASE_URL = os.getenv("DATABASE_URL")
DATABASE_SCHEMA = os.getenv("DATABASE_SCHEMA")
print("DATABASE_URL: ", DATABASE_URL)
print("DATABASE_SCHEMA: ", DATABASE_SCHEMA)

cursor = None

def get_object_images_from_db(object_id):
    """
    Retrieves the images associated with a Residence object from the database.

    :param object_id: The ID of the Residence object.
    :return: A list of image URLs or an empty list if not found.
    """
    try:
        # Establish the database connection
        connection = psycopg2.connect(DATABASE_URL)
        cursor = connection.cursor(cursor_factory=RealDictCursor)
        cursor.execute(f"SET search_path TO {DATABASE_SCHEMA};")

        # Query to fetch the images for the given object ID
        query = """
        SELECT pictures
        FROM "Residence"
        WHERE id = %s;
        """
        cursor.execute(query, (object_id,))
        result = cursor.fetchone()

        # Extract and return the pictures
        return result['pictures'] if result else []

    except Exception as e:
        print(f"Error fetching object images: {e}")
        return []

    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


def get_object_metadata_from_db(object_id):
    """
    Retrieves the metadata associated with a Residence object from the database:
        area        Float
        roomCount   Integer
        elevatorAvailable   Boolean
        floor           Integer
        buildingFloors  Integer

    :param object_id: The ID of the Residence object.
    :return: A dictionary of metadata or an empty dictionary if not found.
    """

    connection = None
    cursor = None

    try:
        # Establish the database connection
        connection = psycopg2.connect(DATABASE_URL)
        cursor = connection.cursor(cursor_factory=RealDictCursor)
        cursor.execute(f"SET search_path TO {DATABASE_SCHEMA};")

        # Query to fetch the metadata for the given object ID
        query = """
        SELECT area, "roomCount", "elevatorAvailable", floor, "buildingFloors"
        FROM "Residence"
        WHERE id = %s;
        """
        cursor.execute(query, (object_id,))
        result = cursor.fetchone()

        # Extract and return the metadata
        return result if result else {}
    except Exception as e:
        print(f"Error fetching object metadata: {e}")
        return {}

    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()



def get_all_residences():
    """
    Retrieves and prints all Residence objects from the database.

    :return: None
    """
    connection = None
    cursor = None

    try:
        # Establish the database connection
        connection = psycopg2.connect(DATABASE_URL)
        cursor = connection.cursor(cursor_factory=RealDictCursor)
        cursor.execute(f"SET search_path TO {DATABASE_SCHEMA};")

        # Query to fetch all Residence objects
        query = "SELECT * FROM \"Residence\";"
        cursor.execute(query)
        results = cursor.fetchall()

        # Print each Residence object
        for residence in results:
            print(residence)

    except Exception as e:
        print(f"Error fetching residences: {e}")

    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()



def update_object_predicted_price_in_db(object_id, predicted_price):
    """
    Updates the estimated price of a Residence object in the database.

    :param object_id: The ID of the Residence object.
    :param predicted_price: Predicted price for the object.
    :return: None
    """
    try:
        # Establish the database connection
        connection = psycopg2.connect(DATABASE_URL)
        cursor = connection.cursor(cursor_factory=RealDictCursor)
        cursor.execute(f"SET search_path TO {DATABASE_SCHEMA};")

        print("All columns of Residence table: ", cursor.execute("SELECT * FROM \"Residence\";"))

        # Query to update the estimated price for the given object ID
        query = """
            UPDATE "Residence"
            SET "predictedPrice" = %s
            WHERE id = %s;
            """
        cursor.execute(query, (predicted_price, object_id))
        connection.commit()

    except Exception as e:
        print(f"Error updating object estimated price: {e}")

    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()