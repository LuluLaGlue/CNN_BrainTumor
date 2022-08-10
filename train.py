import os
import numpy as np
import pandas as pd
from tensorflow import keras
from cv2 import imread, resize
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from tensorflow.keras.callbacks import History
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dense, Flatten,
                                     Input, RandomFlip, RandomZoom,
                                     RandomContrast, RandomRotation)


def import_images(
    categories: list[str, str] = ["CLEAN", "TUMOR"],
    input_path: str = "brain_tumor_dataset",
    img_size: tuple[int, int] = (256, 256)
) -> tuple[tuple, tuple, tuple]:
    '''Given a folder import all images present in the *train*, *test* and *validation* subfolder and returns them with proper label.
    
    Arguments:
        - categories (list): categories of images present in subfolders. Default: ["CLEAN", "TUMOR"]
        - input_path (str): path to the folder to import. Default: "brain_tumor_dataset"
        - img_size (tuple): size to use when resizing images. Default: (256, 256)
    Returns:
        (train_data, train_labels), (test_data, test_labels), (val_data, val_labels)
    '''
    test_X, test_Y = [], []
    train_X, train_Y = [], []
    val_X, val_Y = [], []

    for i, cond in enumerate(categories):
        for _set in ["test", "train", "validation"]:
            for img in os.listdir(os.path.join(input_path, _set, cond)):
                img = imread(os.path.join(input_path, _set, cond, img), 0)
                img = resize(img, img_size)
                img = np.dstack([img, img, img])
                img = img.astype("float32") / 255

                if _set == "test":
                    test_X.append(img)
                    test_Y.append(i)
                elif _set == "train":
                    train_X.append(img)
                    train_Y.append(i)
                elif _set == "validation":
                    val_X.append(img)
                    val_Y.append(i)

    test_X = np.array(test_X)
    test_Y = np.array(test_Y)
    train_X = np.array(train_X)
    train_Y = np.array(train_Y)
    val_X = np.array(val_X)
    val_Y = np.array(val_Y)

    return (train_X, train_Y), (test_X, test_Y), (val_X, val_Y)


def create_model(img_size: tuple[int, int] = (256, 256),
                 should_data_augmentation: bool = False) -> Model:
    '''Create and returns a CNN model using Tensorflow.
    
    Parameters:
        - img_size (tuple): Image size (x, y). Default: (256, 256)
        - should_data_augmentation (boolean): Whether to use data augmentation in model. Default: False
        
    Returns:
        - A compiled Tensorflow CNN model.
    '''
    layers = [
        Input(shape=(img_size[0], img_size[1], 3)),
        Conv2D(16, (3, 3), 1, activation="relu", padding="same"),
        Conv2D(16, (3, 3), 1, activation="relu", padding="same"),
        MaxPooling2D(),
        Conv2D(32, (5, 5), 1, activation="relu", padding="same"),
        Conv2D(32, (5, 5), 1, activation="relu", padding="same"),
        MaxPooling2D(),
        Conv2D(16, (3, 3), 1, activation="relu", padding="same"),
        Conv2D(16, (3, 3), 1, activation="relu", padding="same"),
        MaxPooling2D(),
        Flatten(),
        Dense(128, activation="relu"),
        Dense(1, activation="sigmoid")
    ]

    if should_data_augmentation:
        data_augmentation = Sequential([
            RandomFlip("horizontal_and_vertical",
                       input_shape=(img_size[0], img_size[1], 3)),
            RandomZoom(0.1),
            RandomContrast(0.1),
            RandomRotation(0.2)
        ])
        layers[0] = data_augmentation

    model = Sequential(layers)

    model.compile(optimizer="adam",
                  loss=keras.losses.BinaryCrossentropy(),
                  metrics=["accuracy"])

    model.summary()

    return model


def metrics(hist: History, model: Model, test_data: tuple[list, list],
            epochs: int, name: str) -> str:
    '''Based on a training history, calculates and displays some basic useful metrics.
    
    Parameters:
        - hist (History): Training history.
        - model (Model): Model used during training.
        - test_data (list): Tuple containing both data and labels for testing (data, labels).
        - epochs (int): Number of epochs used during training.
        - name (str): Model custom given name.
    
    Returns:
        - The model name formatted as follow: {epochs}_a{accuracy}_f{F1 Score}_{model number}_{custom name}
    '''
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    ax = ax.ravel()

    for i, met in enumerate(["accuracy", "loss"]):
        ax[i].plot(hist.history[met])
        ax[i].plot(hist.history["val_" + met])

        ax[i].set_title("Model {}".format(met))
        ax[i].set_xlabel("epochs")
        ax[i].set_ylabel(met)
        ax[i].legend(["train", "val"])

    preds = model.predict(test_data[0])

    acc = accuracy_score(test_data[1], np.round(preds)) * 100
    cm = confusion_matrix(test_data[1], np.round(preds))
    _, fp, fn, tp = cm.ravel()

    precision = tp / (tp + fp) * 100
    recall = tp / (tp + fn) * 100
    f = 2 * precision * recall / (precision + recall)

    try:
        acc = int(acc)
    except:
        acc = 0

    try:
        f = int(f)
    except:
        f = 0

    try:
        recall = int(recall)
    except:
        recall = 0

    try:
        precision = int(precision)
    except:
        precision = 0

    print("---------- METRICS ----------")
    print("-----       TEST       -----")
    print("Accuracy: {}%".format(acc))
    print("Precision: {}%".format(precision))
    print("Recall: {}%".format(recall))
    print("F1-Score: {}%".format(f))

    print("-----      TRAIN       -----")
    print("Accuracy: {}%".format(
        np.round((hist.history["accuracy"][-1]) * 100, 2)))

    df = pd.read_csv(os.path.join("models", "models_data.csv"), sep=";")

    size = df.shape[0] + 1

    model_name = os.path.join(
        "models", "{}_a{}_f{}_{}_{}".format(epochs, int(acc), int(f), size,
                                            name)).split(os.sep)
    model_name.pop(0)
    model_name = os.sep.join(model_name)

    new_row = [[
        size, model_name,
        np.round(hist.history["accuracy"][-1] * 100, 2),
        np.round(acc, 2),
        np.round(precision, 2),
        np.round(recall, 2),
        np.round(f, 2)
    ]]

    new_df = pd.DataFrame(new_row, columns=df.columns.values)
    df = df.append(new_df)
    df.set_index('id', drop=True, inplace=True)

    model.save(os.path.join("models", model_name, "model"))
    df.to_csv(os.path.join("models", "models_data.csv"), sep=";")
    plt.savefig(os.path.join("models", model_name, "training_graph.png"))

    return model_name


if __name__ == "__main__":
    parser = ArgumentParser(description="Train CNN")
    parser.add_argument(
        "-i",
        "--input_path",
        type=str,
        default="brain_tumor_dataset",
        help="Path to input images",
    )
    parser.add_argument("-s",
                        "--img_size",
                        type=int,
                        default=256,
                        help="Image size (note that width = height)")
    parser.add_argument("-e",
                        "--epochs",
                        type=int,
                        default=15,
                        help="Number of epochs for training")
    parser.add_argument("-b",
                        "--batch_size",
                        type=int,
                        default=32,
                        help="Batch size")
    parser.add_argument("-w",
                        "--workers",
                        type=int,
                        default=32,
                        help="Number of workers to use during training")
    parser.add_argument("-n",
                        "--name",
                        type=str,
                        default="",
                        help="Name to give to the model.")
    args = parser.parse_args()

    input_path = args.input_path
    img_size = (args.img_size, args.img_size)
    epochs = args.epochs
    batch_size = args.batch_size
    workers = args.workers
    name = args.name

    train, test, val = import_images(categories=["CLEAN", "TUMOR"],
                                     input_path=input_path,
                                     img_size=img_size)
    model = create_model(img_size, False)

    hist = model.fit(x=train[0],
                     y=train[1],
                     epochs=epochs,
                     validation_data=val,
                     batch_size=batch_size,
                     workers=workers)

    model_name = metrics(hist=hist,
                         model=model,
                         test_data=test,
                         epochs=epochs,
                         name=name)