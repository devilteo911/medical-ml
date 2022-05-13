import numpy as np
import pandas as pd
from pathlib import Path
import logging
import cv2
from sklearn.model_selection import train_test_split


def image_shower(img):
    assert isinstance(img, np.ndarray), "img should be of type numpy.ndarray"
    cv2.imshow("img", img)
    cv2.waitKey(0)  # waits until a key is pressed
    cv2.destroyAllWindows()  # destroys the window showing image


def file_loader():
    try:
        idx = np.load("idx.npy")  # indices
        images = np.load("image.npy", allow_pickle=True)  # images with shape (512, 512)
        return idx, images
    except FileNotFoundError as e:
        print(f"{e} occurred. Check if the path is correct")


def set_creator(idx, imgs):
    X_train, X_test, y_train, y_test = train_test_split(
        imgs, idx, test_size=0.1, random_state=42
    )
    return X_train, X_test, y_train, y_test


def main(**kwargs):

    logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
    idx, images = file_loader()
    logging.info("File were successfully loaded!")
    if kwargs["show_img"]:
        image_shower(images[0])
    X_train, X_test, y_train, y_test = set_creator(idx, images)
    logging.info(
        f"Shapes are: {X_train.shape}, {X_test.shape}, {y_train.shape}, {y_test.shape}"
    )


if __name__ == "__main__":
    main(show_img=False)
