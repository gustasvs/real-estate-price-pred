from model import transformer_model
from preprocess_images import preprocess_images


def train_model():
    """
    Function to train the model.
    """

    training_data = preprocess_images()

    model = transformer_model()


    model.fit(
        training_data,
        epochs=10,
        batch_size=32,
    )
    
    return model