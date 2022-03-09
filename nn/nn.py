# BMI 203 Project 7: Neural Network


# Importing Dependencies
import numpy as np
from typing import List, Dict, Tuple, Union
from numpy.typing import ArrayLike
from sklearn.utils import shuffle


# Neural Network Class Definition
class NeuralNetwork:
    """
    This is a neural network class that generates a fully connected Neural Network.

    Parameters:
        nn_arch: List[Dict[str, float]]
            This list of dictionaries describes the fully connected layers of the artificial neural network.
            e.g. [{'input_dim': 64, 'output_dim': 32, 'activation': 'relu'}, {'input_dim': 32, 'output_dim': 8, 'activation:': 'sigmoid'}] will generate a
            2 layer deep fully connected network with an input dimension of 64, a 32 dimension hidden layer
            and an 8 dimensional output.
        lr: float
            Learning Rate (alpha).
        seed: int
            Random seed to ensure reproducibility.
        batch_size: int
            Size of mini-batches used for training.
        epochs: int
            Max number of epochs for training.
        loss_function: str
            Name of loss function.

    Attributes:
        arch: list of dicts
            This list of dictionaries describing the fully connected layers of the artificial neural network.
    """
    def __init__(self,
                 nn_arch,
                 #nn_arch: List[Dict[str, Union(int, str)]],
                 lr: float,
                 seed: int,
                 batch_size: int,
                 epochs: int,
                 loss_function: str):
        # Saving architecture
        self.arch = nn_arch
        # Saving hyperparameters
        self._lr = lr
        self._seed = seed
        self._epochs = epochs
        self._loss_func = loss_function
        self._batch_size = batch_size
        # Initializing the parameter dictionary for use in training
        self._param_dict = self._init_params()

    def _init_params(self) -> Dict[str, ArrayLike]:
        """
        DO NOT MODIFY THIS METHOD!! IT IS ALREADY COMPLETE!!

        This method generates the parameter matrices for all layers of
        the neural network. This function returns the param_dict after
        initialization.

        Returns:
            param_dict: Dict[str, ArrayLike]
                Dictionary of parameters in neural network.
        """
        # seeding numpy random
        np.random.seed(self._seed)
        # defining parameter dictionary
        param_dict = {}
        # initializing all layers in the NN
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            input_dim = layer['input_dim']
            output_dim = layer['output_dim']
            # initializing weight matrices
            param_dict['W' + str(layer_idx)] = np.random.randn(output_dim, input_dim) * 0.1
            # initializing bias matrices
            param_dict['b' + str(layer_idx)] = np.random.randn(output_dim, 1) * 0.1
        return param_dict

    def _single_forward(self,
                        W_curr: ArrayLike,
                        b_curr: ArrayLike,
                        A_prev: ArrayLike,
                        activation: str) -> Tuple[ArrayLike, ArrayLike]:
        """
        This method is used for a single forward pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            activation: str
                Name of activation function for current layer.

        Returns:
            A_curr: ArrayLike
                Current layer activation matrix.
            Z_curr: ArrayLike
                Current layer linear transformed matrix.
        """

        Z_curr = A_prev.dot(W_curr.T) + b_curr.T
        ## Look for the correct activation
        assert (activation == "Relu" or activation == "Sigmoid"),"Not correct activation function"
        if activation == "Sigmoid":
            A_curr = self._sigmoid(Z_curr)
            return (A_curr, Z_curr)
        elif activation == "Relu":
            A_curr = self._relu(Z_curr)
            return (A_curr, Z_curr)

    def forward(self, X: ArrayLike) -> Tuple[ArrayLike, Dict[str, ArrayLike]]:
        """
        This method is responsible for one forward pass of the entire neural network.

        Args:
            X: ArrayLike
                Input matrix with shape [batch_size, features].

        Returns:
            output: ArrayLike
                Output of forward pass.
            cache: Dict[str, ArrayLike]:
                Dictionary storing Z and A matrices from `_single_forward` for use in backprop.
        """

        ##Initialize the cache
        cache = {}

        ## add the output as first element
        cache['A0']= X
        A_prev = X

        # We can use the same logic as contrusting the network
        for i in range(len(self.arch)):
            i_index = i + 1
            ## get the current matrices from the dictionary
            W_curr = self._param_dict['W' + str(i_index)]
            b_curr = self._param_dict['b' + str(i_index)]
            activation = self.arch[i_index-1]["activation"]
            ## Calculate A and Z
            A_curr, Z_curr = self._single_forward(W_curr, b_curr, A_prev, activation)
            ## Add current A values
            cache['A' + str(i_index)] = A_curr
            # add Z values into the cache
            cache['Z' + str(i_index)] = Z_curr
        ## A_curr at the end will be my output
            A_prev = A_curr
        return A_curr, cache



    def _single_backprop(self,
                         W_curr: ArrayLike,
                         b_curr: ArrayLike,
                         Z_curr: ArrayLike,
                         A_prev: ArrayLike,
                         dA_curr: ArrayLike,
                         activation_curr: str) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        This method is used for a single backprop pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            Z_curr: ArrayLike
                Current layer linear transform matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            dA_curr: ArrayLike
                Partial derivative of loss function with respect to current layer activation matrix.
            activation_curr: str
                Name of activation function of layer.

        Returns:
            dA_prev: ArrayLike
                Partial derivative of loss function with respect to previous layer activation matrix.
            dW_curr: ArrayLike
                Partial derivative of loss function with respect to current layer weight matrix.
            db_curr: ArrayLike
                Partial derivative of loss function with respect to current layer bias matrix.


        Reference : https://mlfromscratch.com/neural-networks-explained/#/

        """
        ## We use the derivate of the activation function
        ## Which again, is the multiplication of dL_dA * dL_dZ
        if activation_curr == "Sigmoid":
            dL_dA_dL_dZ = self._sigmoid_backprop(dA_curr,Z_curr)
        elif activation_curr == "Relu":
            dL_dA_dL_dZ =self._relu_backprop(dA_curr,Z_curr)

        ## dLoss/dWeight = dL_dA * dL_dZ * A[L-1]
        dW_curr = dL_dA_dL_dZ.T.dot(A_prev)
        
        ## dLoss / dA[-1] = dL_dA * dL_dZ * weight
        dA_prev = dL_dA_dL_dZ.dot(W_curr)
        
        #dLoss/ dBias = dL_dA * dL_dZ * 1
        #db_curr = dL_dA_dL_dZ * 1
        db_curr = np.sum((dL_dA_dL_dZ), axis=0)
        db_curr = db_curr.reshape(b_curr.shape)

        return dA_prev, dW_curr, db_curr

    def backprop(self, y: ArrayLike, y_hat: ArrayLike, cache: Dict[str, ArrayLike]):
        """
        This method is responsible for the backprop of the whole fully connected neural network.

        Args:
            y (array-like):
                Ground truth labels.
            y_hat: ArrayLike
                Predicted output values.
            cache: Dict[str, ArrayLike]
                Dictionary containing the information about the
                most recent forward pass, specifically A and Z matrices.

        Returns:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from this pass of backprop.
        """
        ## Initialize the gradient directory
        grad_dict = {}

        ## Get the corresponding loss function

        dA_curr_loss = self._loss_function_backdrop(y,y_hat)
        dA_curr = dA_curr_loss
        ##check this
        for i in reversed(range(len(self.arch))):
            index_L = i + 1
            ## Do a single back drop
            dA_prev, dW_curr, db_curr = self._single_backprop(
                         W_curr = self._param_dict['W'+ str(index_L)],
                         b_curr = self._param_dict['b'+ str(index_L)],
                         Z_curr = cache['Z'+ str(index_L)],
                         A_prev = cache['A'+ str(index_L - 1)], ## remember is the previous 
                         dA_curr = dA_curr,
                         activation_curr = self.arch[index_L-1]["activation"])

            dA_curr = dA_prev ## Update for next pass
            ## Store in ditionary
            grad_dict['W' + str(index_L)] = dW_curr
            grad_dict['b' + str(index_L)] = db_curr


        return(grad_dict)

    def _update_params(self, grad_dict: Dict[str, ArrayLike]):
        """
        This function updates the parameters in the neural network after backprop. This function
        only modifies internal attributes and thus does not return anything

        Args:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from most recent round of backprop.

        Returns:
            None
        """
        for idx in (range(len(self.arch))):

            idx = idx + 1 
            self._param_dict["W" + str(idx)]= self._lr * self._param_dict["W" + str(idx)] -  grad_dict["W" + str(idx)] 
            self._param_dict["b" + str(idx)]= self._lr * self._param_dict["b" + str(idx)] - grad_dict["b" + str(idx)]


    def fit(self,
            X_train: ArrayLike,
            y_train: ArrayLike,
            X_val: ArrayLike,
            y_val: ArrayLike) -> Tuple[List[float], List[float]]:
        """
        This function trains the neural network via training for the number of epochs defined at
        the initialization of this class instance.
        Args:
            X_train: ArrayLike
                Input features of training set.
            y_train: ArrayLike
                Labels for training set.
            X_val: ArrayLike
                Input features of validation set.
            y_val: ArrayLike
                Labels for validation set.

        Returns:
            per_epoch_loss_train: List[float]
                List of per epoch loss for training set.
            per_epoch_loss_val: List[float]
                List of per epoch loss for validation set.

        Note: I used the some parts of the fit method from  project 6  

        """
        iteration = 0
        per_epoch_loss_train = []
        per_epoch_loss_val = []

        while iteration < self._epochs:
            # Shuffling the training data and dividing in batch
            X_batch,y_batch =  self._random_batch(X_train,y_train)

            ## Save the batch specific loss
            batch_train_L=[]
            batch_validation_L=[]

            #Iterating through batches
            for X_train, y_train in zip(X_batch, y_batch):
                # Making prediction on batch
                y_pred,cache = self.forward(X_train)
                # Calculating loss using the according loss function
                loss_train = self._loss_function(y_train,y_pred)
                # Add the value to loss history record
                batch_train_L.append(loss_train)
                ## Now we do the back grop
                grad_dict = self.backprop(y_train, y_pred, cache)
                self._update_params(grad_dict) ## Update using the derivates
                final_pred, _ = self.predict(X_val) ## check the validation data 
                # Validation pass
                loss_val = self._loss_function(y_val, final_pred)
                batch_validation_L.append(loss_val)
                ## Now we update  the epoch loss with the mean
            per_epoch_loss_train.append(np.mean(batch_train_L))
            per_epoch_loss_val.append(np.mean(batch_validation_L))

            # Updating iteration number
            iteration += 1
        return per_epoch_loss_train,per_epoch_loss_val




    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        This function returns the prediction of the neural network model.

        Args:
            X: ArrayLike
                Input data for prediction.

        Returns:
            y_hat: ArrayLike
                Prediction from the model.
        """
        y_hat, cache = self.forward(X)
        return y_hat, cache

    def _sigmoid(self, Z: ArrayLike) -> ArrayLike:
        """
        Sigmoid activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        ## according to the formula
        return 1.0/(1 + np.e**(-Z))

    def _relu(self, Z: ArrayLike) -> ArrayLike:
        """
        ReLU activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        ## based on the shape of the relu visualization
        ## reference 
        # https://medium.com/@kanchansarkar/relu-not-a-differentiable-function-why-used-in-gradient-based-optimization-7fef3a4cecec
        return np.maximum(0,Z)

    def _sigmoid_backprop(self, dA: ArrayLike, Z: ArrayLike):
        """
        Sigmoid derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        ## Derivate of sigmoid activation
        sigmoid_act= self._sigmoid(Z)*(1-self._sigmoid(Z))
        dZ = dA * sigmoid_act
        return dZ

    def _relu_backprop(self, dA: ArrayLike, Z: ArrayLike) -> ArrayLike:
        """
        ReLU derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        # if Z < 0 == 0. if Z == 0 == 1.
        # The derivative f '(0) is not defined. So 1
        Z = np.where(Z == 0, 1, Z )
        d = np.where(Z > 0, Z, 0)
        dZ = dA * d

        return dZ

    def _binary_cross_entropy(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Binary cross entropy loss function.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            loss: float
                Average loss over mini-batch.
        """
         ## check we might need epsilon
         ## this is for the sigmoid like function 
        loss =  -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)).mean()

        return loss

    def _binary_cross_entropy_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Binary cross entropy loss function derivative.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        ## check this 
        m = len(y)
        dA = ((1-y)/(1- y_hat)-(y/y_hat))/m

        return dA

    def _mean_squared_error(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Mean squared error loss.

        Args:
            y: ArrayLike
                Ground truth output.
            y_hat: ArrayLike
                Predicted output.

        Returns:
            loss: float
                Average loss of mini-batch.
        """
        ## Following the formula
        loss = np.mean((y - y_hat)**2)
        return loss

    def _mean_squared_error_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Mean square error loss derivative.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        m = len(y)
        error = y - y_hat
        dA = -(2*(error)/m)

        return dA

    def _loss_function_backdrop(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        This function choose between the selected loss function
        Args:
            y: ArrayLike
                Ground truth output.
            y_hat: ArrayLike
                Predicted output.
        Returns:
            loss: float
                Average loss of mini-batch.
        """
        if self._loss_func == 'mse':
            dA_curr = self._mean_squared_error_backprop(y, y_hat)
        else:
            dA_curr = self._binary_cross_entropy_backprop(y, y_hat)
            
        return dA_curr

    def _loss_function(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        This function choose between the selected loss function
        Args:
            y: ArrayLike
                Ground truth output.
            y_hat: ArrayLike
                Predicted output.
        Returns:
            loss: float
                Average loss of mini-batch.
        """
        if self._loss_func == 'mse':
            dA_curr = self._mean_squared_error(y, y_hat)
        else:
            dA_curr = self._binary_cross_entropy(y, y_hat)
            
        return dA_curr

    def _random_batch(self, X_train: ArrayLike,
            y_train: ArrayLike):
        """
         Function to randomiize and split data 
        """
        X_train, y_train = shuffle(X_train, y_train)

        ## Defining number of batches
        num_batches = int(X_train.shape[0]/(self._batch_size)) + 1
        X_batch = np.array_split(X_train, num_batches)
        y_batch = np.array_split(y_train, num_batches)

        return(X_batch,y_batch)

