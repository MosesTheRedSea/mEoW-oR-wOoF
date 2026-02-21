import os
import h5py
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

def parse_train_dataset(directory, img_size=64, max_num_=None):
    files = [f for f in os.listdir(directory) if f.lower().endswith(".jpg")]
    files.sort()

    if max_num_ is not None:
        files = files[:max_num_]

    X, Y = [], []

    for file in tqdm(files, desc="Loading train images", unit="img"):
        if file.startswith("cat."):
            y = 0
        elif file.startswith("dog."):
            y = 1
        else:
            continue

        image = Image.open(os.path.join(directory, file)).convert("RGB")
        image = image.resize((img_size, img_size))
        X.append(np.asarray(image, dtype=np.uint8))
        Y.append(y)

    if len(X) == 0:
        raise ValueError(f"No labeled images found in {directory}. Expected cat.*.jpg or dog.*.jpg")

    X = np.stack(X, axis=0)
    Y = np.array(Y, dtype=np.uint8).reshape(1, -1)
    classes = np.array([b"cat", b"dog"])

    return X, Y, classes


def parse_test_dataset(directory, img_size=64, max_num_=None):
    files = [f for f in os.listdir(directory) if f.lower().endswith(".jpg")]

    # Kaggle test filenames are "1.jpg", "2.jpg", ...
    files.sort(key=lambda f: int(os.path.splitext(f)[0]))

    if max_num_ is not None:
        files = files[:max_num_]

    X, ids = [], []

    for file in tqdm(files, desc="Loading test images", unit="img"):
        ids.append(int(os.path.splitext(file)[0]))
        image = Image.open(os.path.join(directory, file)).convert("RGB")
        image = image.resize((img_size, img_size))
        X.append(np.asarray(image, dtype=np.uint8))

    if len(X) == 0:
        raise ValueError(f"No .jpg images found in {directory}")

    X = np.stack(X, axis=0)
    return X, np.array(ids)

    

def load_dataset(train_directory, test_directory, img_size=64, max_train=None, max_test=None):

    train_X, train_Y, classes = parse_train_dataset(train_directory, img_size=img_size, max_num_=max_train)

    test_X, test_ids = parse_test_dataset(test_directory, img_size=img_size, max_num_=max_test)

    test_Y = None  # Kaggle test has no labels

    return train_X, train_Y, test_X, test_Y, classes, test_ids
    