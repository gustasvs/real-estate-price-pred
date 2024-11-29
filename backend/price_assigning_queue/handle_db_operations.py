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
