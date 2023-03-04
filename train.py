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
    class_idx = np.where(train_label == i)[0][0]
    axs[row, col].imshow(train_image[class_idx], cmap='gray')
    axs[row, col].set_title(class_type[i])
    axs[row, col].axis('off')
plt.show()
train_image = train_image.reshape(train_image.shape[0], train_image.shape[1]*train_image.shape[2])
train_label = train_label.reshape(train_label.shape[0],1)