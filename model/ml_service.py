import json
import os
import time

import numpy as np
import redis
import settings

# import settings.py
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image

# TODO ✅
# Connect to Redis and assign to variable `db``
# Make use of settings.py module to get Redis settings like host, port, etc.
db = redis.Redis(
    host=settings.REDIS_IP, port=settings.REDIS_PORT, db=settings.REDIS_DB_ID
)

# TODO ✅
# Load your ML model and assign to variable `model`
# See https://drive.google.com/file/d/1ADuBSE4z2ZVIdn66YDSwxKv-58U7WEOn/view?usp=sharing
# for more information about how to use this model.
# model = None

model = ResNet50(include_top=True, weights="imagenet")


def predict(image_list):
    """
    Load image from the corresponding folder based on the image name
    received, then, run our ML model to get predictions.

    Parameters
    ----------
    image_name : str
        Image filename.

    Returns
    -------
    class_name, pred_probability : tuple(str, float)
        Model predicted class as a string and the corresponding confidence
        score as a number.
    """
    class_name = None
    pred_probability = None

    # TODO ✅
    # We need to convert the PIL image to a Numpy
    # array before sending it to the model

    img_array = np.zeros((len(image_list), 224, 224, 3))

    for number, image_name in enumerate(image_list, start=0):
        img = image.load_img(
            f"{settings.UPLOAD_FOLDER}/{image_name}", target_size=(224, 224)
        )

        x = image.img_to_array(img)
        img_array[number] = x

    # Also we must add an extra dimension to this array
    # because our model is expecting as input a batch of images.
    # In this particular case, we will have a batch with a single
    # image inside
    x_batch = preprocess_input(img_array)

    # Run model on batch of images
    predictions = model.predict(x_batch)

    predictions_decoded = decode_predictions(predictions, top=1)

    class_name_list = []
    pred_probability_list = []

    for prediction in predictions_decoded:
        _, class_name, pred_probability = prediction[0]
        class_name_list.append(class_name)
        pred_probability_list.append(round(pred_probability, 4))

    return class_name_list, pred_probability_list


def classify_process():
    """
    Loop indefinitely asking Redis for new jobs.
    When a new job arrives, takes it from the Redis queue, uses the loaded ML
    model to get predictions and stores the results back in Redis using
    the original job ID so other services can see it was processed and access
    the results.

    Load image from the corresponding folder based on the image name
    received, then, run our ML model to get predictions.
    """
    while True:
        # Inside this loop you should add the code to:
        #   1. Take a new job from Redis
        #   2. Run your ML model on the given data
        #   3. Store model prediction in a dict with the following shape:
        #      {
        #         "prediction": str,
        #         "score": float,
        #      }
        #   4. Store the results on Redis using the original job ID as the key
        #      so the API can match the results it gets to the original job
        #      sent
        # Hint: You should be able to successfully implement the communication
        #       code with Redis making use of functions `brpop()` and `set()`.
        # TODO ✅
        # Take a new job from Redis
        # Checking the queue with name settings.REDIS_QUEUE
        data = db.rpop(settings.REDIS_QUEUE, count=20)

        # Converting the JSON from job_data to a Dict
        if data:
            jobid_list = []
            jobimg_list = []

            # Creates a list of jobs and its names
            for job in data:
                # Loads data as dict
                msg = json.loads(job)

                # If both exists, then predict

                image_name, job_id = msg["image_name"], msg["id"]
                if (image_name, job_id) is not None:
                    jobid_list.append(job_id)
                    jobimg_list.append(image_name)

                else:
                    print("Something went wrong with one job id, please try again")
                # Sending results to redis hashtable

            if len(jobimg_list) > 0:
                class_pred_list, pred_proba_list = predict(jobimg_list)

                ##Send back inidvidual jobs to hash table
                for class_name, pred_probability, job_id in zip(
                    class_pred_list, pred_proba_list, jobid_list
                ):
                    msg_content = {
                        "prediction": class_name,
                        "score": eval(str(pred_probability)),
                    }

                    # Turning msg content into a JSON
                    prediction_content = json.dumps(msg_content)

                    # Sending the message
                    db.set(job_id, prediction_content)

        # Sleep for a bit
        time.sleep(settings.SERVER_SLEEP)


if __name__ == "__main__":
    # Now launch process
    print("Launching ML service...")
    classify_process()
