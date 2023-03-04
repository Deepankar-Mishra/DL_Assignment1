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

class FNN():
    def __init__(self, size_of_each_layer, no_of_layer, epochs=10, L_rate=0.001, optm='sgd', batch_size=16, act_f='sigmoid', loss_f='cross_entropy', output_act_f='softmax', initializer='xavier'):
        self.size_of_each_layer = size_of_each_layer			
        self.no_of_layer = no_of_layer
        self.epochs = epochs
        self.L_rate = L_rate
        self.optm = optm                #Optimizer
        self.batch_size = batch_size
        self.act_f = act_f
        self.initializer = initializer		#Weight inititialisation
        self.loss_f = loss_f	
        self.output_act_f = output_act_f	
        self.parameters = self.parameter_initialisation()	#Weights and biases initialisation.
        
    def activation_function(self, x,output_layer=0, return_derivative=0):
        #IF OUTPUT LAYER =>
        if output_layer:
            exps = np.exp(x - x.max())
            if return_derivative:
                return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
            return exps / np.sum(exps, axis=0)
        #IF NOT OUTPUT LAYER GO BELOW =>    
        if self.act_f == 'sigmoid':
            if return_derivative:
                return (np.exp(-x))/((np.exp(-x)+1)**2)
            else:
                return 1/(1 + np.exp(-x))
        elif self.act_f == 'tanh':
            result= (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
            if return_derivative:
                return 1 - (result)*(result)
            else:
                return result
        elif self.act_f == 'relu':
            if return_derivative:
                return 1. * (x > 0)
            return x * (x > 0)
    def parameter_initialisation(self):
        parameters = {}
        for level in range(1, self.no_of_layer):
            if self.initializer == 'random':
                parameters["W" + str(level)] = np.random.randn(self.size_of_each_layer[level], self.size_of_each_layer[level - 1])*0.1
            elif self.initializer == 'xavier':
                parameters["W" + str(level)] = np.random.randn(self.size_of_each_layer[level], self.size_of_each_layer[level - 1]) * np.sqrt(2/ (self.size_of_each_layer[level - 1] + self.size_of_each_layer[level]))
            parameters["b" + str(level)] = np.zeros((self.size_of_each_layer[level], 1))
        return parameters