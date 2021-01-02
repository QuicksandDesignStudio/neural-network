import os
import LibAccess
import numpy as np

from lib.tools.Data_Operations import Loading_Operations, Useful_Operations
from lib.Models import Model

fashion_mnist_labels = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot",
}

X_test, y_test = Loading_Operations.get_testing_data_only(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "fashion_mnist_images")
)

shuffled_keys = Useful_Operations.shuffle_keys(X_test)
X_test = X_test[shuffled_keys]
y_test = y_test[shuffled_keys]

X_test = Useful_Operations.linear_scaling_negative1_positive1(X_test, 127.5, np.float32)

X_test = X_test.reshape(X_test.shape[0], -1)

model = Model.load("fashion_minst.model")
confidences = model.predict(X_test[:5])
predictions = model.output_layer_activation.predictions(confidences)

for prediction in predictions:
    print(fashion_mnist_labels[prediction])
