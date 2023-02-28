import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist

# Load the Fashion-MNIST dataset
(train_image,train_label),(test_image,test_label)=fashion_mnist.load_data()

# Define the class names
class_type=['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker', 'Bag','Ankle boot']

# Plot one sample image for each class
fig,axs=plt.subplots(nrows=2, ncols=5, figsize=(12, 6))
for i in range(len(class_type)):
    row=i//5
    col=i%5
    class_idx = np.where(train_labels == i)[0][0]
    axs[row, col].imshow(train_images[class_idx], cmap='gray')
    axs[row, col].set_title(class_type[i])
    axs[row, col].axis('off')
plt.show()

