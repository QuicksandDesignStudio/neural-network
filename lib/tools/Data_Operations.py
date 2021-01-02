import numpy as np
import cv2
import os


class Loading_Operations:
    def __init__(self):
        pass

    @staticmethod
    def load_dataset(dataset, path):
        labels = os.listdir(os.path.join(path, dataset))
        X = []
        y = []

        for label in labels:
            for file in os.listdir(os.path.join(path, dataset, label)):
                image = cv2.imread(
                    os.path.join(path, dataset, label, file), cv2.IMREAD_UNCHANGED
                )
                X.append(image)
                y.append(label)
        return np.array(X), np.array(y).astype("uint8")

    @staticmethod
    def create_folder_labelled_data(path):
        X, y = Loading_Operations.load_dataset("train", path)
        X_test, y_test = Loading_Operations.load_dataset("test", path)
        return X, y, X_test, y_test

    @staticmethod
    def get_testing_data_only(path):
        X_test, y_test = Loading_Operations.load_dataset("test", path)
        return X_test, y_test


class Useful_Operations:
    @staticmethod
    def linear_scaling_negative1_positive1(data, halfValue, type=np.float32):
        return (data.astype(type) - halfValue) / halfValue

    @staticmethod
    def shuffle_keys(refernce_array):
        keys = np.array(range(refernce_array.shape[0]))
        np.random.shuffle(keys)
        return keys
