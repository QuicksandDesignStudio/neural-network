import LibAccess
import cv2
import matplotlib.pyplot as plt
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

image_data = cv2.imread("samples/2.png", cv2.IMREAD_GRAYSCALE)
image_data = cv2.resize(image_data, (28, 28))
image_data = 255 - image_data  # the training set is color inverted
plt.imshow(image_data, cmap="gray")

image_data = Useful_Operations.linear_scaling_negative1_positive1(
    image_data, 127.5, np.float32
)
image_data = image_data.reshape(1, -1)


model = Model.load("fashion_minst.model")
confidences = model.predict(image_data)
predictions = model.output_layer_activation.predictions(confidences)
prediction = fashion_mnist_labels[predictions[0]]
print(prediction)


plt.show()
