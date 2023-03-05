import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
import math
from tqdm import tqdm

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

    def grad_initialisation(self):
        grad={}
        for level in range(1, self.no_of_layer):
            grad["W" + str(level)] = np.zeros((self.size_of_each_layer[level], self.size_of_each_layer[level - 1]))
            grad["b" + str(level)] = np.zeros((self.size_of_each_layer[level], 1))
        return grad
    def Forward_Prop(self, x):
        pre_actn_values = {}
        post_actn_values = {}

        post_actn_values['h0'] = x.reshape(len(x),1)

        # From layer 1 to last_layer-1
        for level in range(1, self.no_of_layer-1):
            pre_actn_values['a' + str(level)] = self.parameters['b' + str(level)] + np.matmul(self.parameters['W' + str(level)], post_actn_values['h' + str(level-1)])
            post_actn_values['h' + str(level)] = self.activation_function(pre_actn_values['a' + str(level)],output_layer=0, return_derivative=0)

        # Last layer
        pre_actn_values['a' + str(self.no_of_layer-1)] = self.parameters['b' + str(self.no_of_layer-1)] + + np.matmul(self.parameters['W' + str(self.no_of_layer-1)], post_actn_values['h' + str(self.no_of_layer-1-1)])
        post_actn_values['h' + str(self.no_of_layer-1)] = self.activation_function(pre_actn_values['a' + str(self.no_of_layer-1)],output_layer=1,return_derivative=0)

        return post_actn_values, pre_actn_values
    #Calculate the value of loss.
    def loss_value(self, y_h, y):
        if self.loss_f == 'squared_error':
            loss_val = 0.5*np.sum((y-y_h)*(y-y_h))
        elif self.loss_f == 'cross_entropy':
            Class_i = np.argmax(y)
            probability_value = y_h[Class_i]
            #Add small positive quantity to probability_value to avoid zero value.
            if(probability_value<=0):
                probability_value = probability_value + 0.00000001		
            loss_val = -math.log(probability_value)
        return loss_val
    def Back_Prop(self, y, post_actn_values, pre_actn_values):
        #Assume forward propagation has been done. Hence Calculate a and h at each level.
        #Step1. Compute output gradient
        #Step2. Traverse each level from top to bottom layer.
        #Step 2.1 --- Calculate W and b for hidden layers.
        #Step 2.2 --- Calculate gradient of hidden layer (Activation h).
        #STep 2.3 --- Calculate gradient of hidden layer (Pre activation a).
        
        grad={}
        f_x=post_actn_values['h' + str(self.no_of_layer-1)]
        e_y = y.reshape(len(y), 1)
        
        # Step 1
        if self.loss_f == 'cross_entropy':
            grad['a' + str(self.no_of_layer-1)] = (f_x - e_y)
        elif self.loss_f == 'squared_error':
            grad['a' + str(self.no_of_layer-1)] = (f_x - e_y)*f_x*(1-f_x)

        # Step 2
        for level in range(self.no_of_layer-1, 0, -1):
            #Step 2.1
            #Reshape the gradients into appropriate shapes for matrix multiplication
            grad_a = grad['a' + str(level)]
            grad_h = post_actn_values['h' + str(level-1)]
            grad['W' + str(level)] = grad_a[:, np.newaxis] @ grad_h[np.newaxis, :]
            grad['b' + str(level)] = grad['a' + str(level)]

            #Step 2.2
            grad['h' + str(level-1)] = self.parameters['W' + str(level)].transpose() @ grad['a' + str(level)]

            #Step 2.3
            if level >= 2:
                grad['a' + str(level-1)] = grad['h' + str(level-1)] * self.activation_function(pre_actn_values['a' + str(level-1)], return_derivative=1)	

        return grad
    def calculate_performance(self, x_test, y_test):
        correct_predictions = 0
        y_true = []
        y_pred = []
        losses = []
        
        for i in tqdm(range(len(x_test))):
            x = x_test[i]
            y = y_test[i]
            
            post_actn_values,pre_actn_values = self.Forward_Prop(x)
            predicted_class = np.argmax(post_actn_values['h' + str(self.L-1)])
            actual_class = np.argmax(y)
            
            y_true.append(actual_class)
            y_pred.append(predicted_class)
            losses.append(self.loss_value(post_actn_values['h' + str(self.L-1)], y))
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        correct_predictions = np.sum(np.equal(y_true, y_pred))
        
        accuracy = (correct_predictions / len(x_test)) * 100
        loss = np.mean(losses)
        
        return accuracy, loss, y_true, y_pred
    def SGD(self,x_train, y_train, x_val, y_val, epochs):
        # Loop over the specified number of epochs
        for epoch in range(epochs):
            print("Epoch ---- ",epoch)
            # Loop over the training examples
            for i in tqdm(range(len(x_train)), total=len(x_train)):
                #Initialise grads
                init_grad=self.grad_initialisation()
                
                x = x_train[i]
                y = y_train[i]            
                # Forward propagation implemented
                post_actn_values, pre_actn_values = self.Forward_Prop(x)
                
                # Backward propagation implemented
                new_Grad = self.Back_Prop(y, post_actn_values, pre_actn_values, self.parameters)
                
                #New value of all parameters
                self.parameters = {key: self.parameters[key] - self.L_rate*new_Grad[key] for key in self.parameters}

                
            #Calculate losses and accuracy for train and validation set.
            Accuracy_Train, Loss_Train = self.calculate_performance(x_train, y_train)
            Accuracy_validation, Loss_validation = self.calculate_performance(x_val, y_val)
            
            # Training and validation accuracy and loss
            print("Training Accuracy :- ", Accuracy_Train)
            print("Validation Accuracy :- ", Accuracy_validation)
            print("Training Loss :- ", Loss_Train)
            print("Validation Loss :- ", Loss_validation)
            
        return