from os import replace, listdir, makedirs
from argparse import ArgumentParser
from os.path import isdir, join
from random import shuffle


def train_test_split(folder: str, train_size: float, output: str) -> None:
    '''Moves images into a test and train folder based on a given proportion.
    
    Parameters:
        - folder: The folder in which the image currently are,
        - train_size: The proportion of images to save for training (0 < train_size < 1),
    '''
    if train_size > 1 or train_size < 0:
        print("PARAM ERROR - train size must between 0 and 1")

        return

    if not isdir(folder):
        print("PARAM ERROR - folder must be a valid folder")

        return
    for step in ["train", "test"]:
        if not isdir(join(output, step)):
            makedirs(join(output, step, "CLEAN"))
            makedirs(join(output, step, "TUMOR"))

    for cond in ["CLEAN", "TUMOR"]:
        path = join(folder, cond)

        images = listdir(path)
        shuffle(images)
        index_separation = int(len(images) * (1 - train_size))

        for index, value in enumerate(images):
            if not isdir(join(path, value)):

                replace(
                    join(path, value),
                    join(output,
                         "test" if index < index_separation else "train", cond,
                         value))

    return


if __name__ == "__main__":
    parser = ArgumentParser(description="Prepare Train/Test Folders")
    parser.add_argument("-p",
                        "--proportion",
                        type=int,
                        default=90,
                        help="Training set proportion in percent")
    parser.add_argument("-i",
                        "--images",
                        type=str,
                        default="images",
                        help="Path to images to split")
    parser.add_argument("-o",
                        "--output",
                        type=str,
                        default="images",
                        help="Output Folder")

    args = parser.parse_args()

    train_test_split(args.images, args.proportion / 100, args.output)