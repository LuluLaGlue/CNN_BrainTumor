from tensorflow.keras.models import load_model
from argparse import ArgumentParser
from train import import_images
import matplotlib.pyplot as plt
import numpy as np
import random
import os

if __name__ == "__main__":
    parser = ArgumentParser(description="Train CNN")
    parser.add_argument(
        "-i",
        "--input_path",
        type=str,
        default="brain_tumor_dataset",
        help="Path to input images",
    )
    parser.add_argument("-m",
                        "--model",
                        type=str,
                        default="15_a76_f80_1_REFERENCE",
                        help="Model's name")
    parser.add_argument("-n",
                        "--number",
                        type=int,
                        default=10,
                        help="Number of images to classify")
    args = parser.parse_args()

    model_name = args.model
    input_path = args.input_path
    nbr = args.number

    model = load_model(os.path.join("models", model_name, "model"))
    train, test, val = import_images(categories=["CLEAN", "TUMOR"],
                                     input_path=input_path)

    correct = 0

    for i in range(nbr - 1):
        index = random.randrange(0, len(test[0]) - 1)
        img = test[0][index]
        label = test[1][index]

        plt.imshow(img)
        if label == 1:
            plt.title("Brain with Tumor")
        else:
            plt.title("Brain with No Tumor")
        plt.show()
        y_hat = model.predict(np.expand_dims(img, 0))

        if y_hat < 0.6:
            print("No tumor detected", 1 - y_hat)
            if label == 0:
                correct += 1
        else:
            print("Tumor detected", y_hat)
            if label == 1:
                correct += 1

    print("Number of correct classifications: {}/{}".format(correct, nbr))
