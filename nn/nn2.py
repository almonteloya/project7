   
# BMI 203 Project 7: Neural Network


# Importing Dependencies
import numpy as np
from typing import List, Dict, Tuple, Union
from numpy.typing import ArrayLike


# Neural Network Class Definition
class NeuralNetwork:
    """
    This is a neural network class that generates a fully connected Neural Network.
    Parameters:
        nn_arch: List[Dict[str, float]]
            This list of dictionaries describes the fully connected layers of the artificial neural network.
            e.g. [{'input_dim': 64, 'output_dim': 32}, {'input_dim': 32, 'output_dim': 8}] will generate a
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
                 nn_arch: List[Dict[str, Union[int, str]]],
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


    def _set_params_for_test(self, new_params: Dict[str, ArrayLike]): 
        """
        This function is only to be used by pytest in order to set initial weights and biases.
        """
        self._param_dict = new_params


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
            Z_curr: ArrayLikelo
                Current layer linear transformed matrix.
        """

        Z_curr = A_prev @ W_curr.T + b_curr.T

        # Calculate activation matrix
        assert (activation in ["sigmoid", "relu"]), "Activation function unrecognized"
        if activation == "sigmoid":
            A_curr = self._sigmoid(Z_curr)
            return (A_curr, Z_curr)
        else: # activation == "relu" 
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
        # Intialize dictionary to store Z, A matrices
        cache = {}

        # Intialize NN input
        cache['A0'] = X
        A_prev = X

        # Move through layers of NN
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            W_curr = self._param_dict['W' + str(layer_idx)]
            b_curr = self._param_dict['b' + str(layer_idx)]
            activation = layer["activation"]

            A_curr, Z_curr = self._single_forward(W_curr, b_curr, A_prev, activation)

            # Store Z and A matrices
            cache['Z' + str(layer_idx)] = Z_curr
            cache['A' + str(layer_idx)] = A_curr

            # Set up input for next layer
            A_prev = A_curr

        # Output is the value of the last layer's activation
        output = A_prev
        return (output, cache)




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
        """

        dL_dAcurr = dA_curr
        dL_dZcurr = []

        assert (activation_curr in ["sigmoid", "relu"]), "Activation function unrecognized"
        if activation_curr == "sigmoid":
            dL_dZcurr = self._sigmoid_backprop(dL_dAcurr, Z_curr)
        else: # activation == "relu" 
            dL_dZcurr = self._relu_backprop(dL_dAcurr, Z_curr)

        num_datapoints = dL_dZcurr.shape[0]

        dL_dAprev = dL_dZcurr @ W_curr                          # n X input
        dL_dWcurr = dL_dZcurr.T @ A_prev                        # output X input
        dL_dbcurr = dL_dZcurr.T @ np.ones((num_datapoints, 1))  # output X 1

        dA_prev = dL_dAprev
        dW_curr = dL_dWcurr
        db_curr = dL_dbcurr

        return(dA_prev, dW_curr, db_curr)

        

    def backprop(self, y: ArrayLike, y_hat: ArrayLike, cache: Dict[str, ArrayLike]) -> Dict[str, ArrayLike]:
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
        # Intialize dictionary to store gradients
        grad_dict = {}

        # Intialize first derivative (dL/dy_hat)
        dL_dy_hat = []
        assert (self._loss_func in ["mse", "bce"]), "Loss function unrecognized"
        if self._loss_func == "mse":
            dL_dy_hat = self._mean_squared_error_backprop(y, y_hat)
        else: # self._loss_func == "bce"
            dL_dy_hat = self._binary_cross_entropy_backprop(y, y_hat)

        dA_curr = dL_dy_hat

        # Move through layers of NN, backwards
        for idx, layer in reversed(list(enumerate(self.arch))):
            layer_idx = idx + 1
            W_curr = self._param_dict['W' + str(layer_idx)]
            b_curr = self._param_dict['b' + str(layer_idx)]
            Z_curr = cache['Z' + str(layer_idx)]
            A_prev = cache['A' + str(layer_idx-1)]
            activation_curr = layer["activation"]

            dA_prev, dW_curr, db_curr = self._single_backprop(W_curr, b_curr, Z_curr, A_prev, dA_curr, activation_curr)

            # Store gradient matrices
            grad_dict['dW' + str(layer_idx)] = dW_curr
            grad_dict['db' + str(layer_idx)] = db_curr

            # Set up dL/dA for next layer
            dA_curr = dA_prev

        # Return gradient dictionary
        return grad_dict




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
        # Update weights and biases for each layer of NN
        # Update strategy: gradient descent
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            W_prev = self._param_dict['W' + str(layer_idx)]
            b_prev = self._param_dict['b' + str(layer_idx)]
            dW = grad_dict['dW' + str(layer_idx)]
            db = grad_dict['db' + str(layer_idx)]
            W_new = W_prev - self._lr * dW 
            b_new = b_prev - self._lr * db 

            self._param_dict['W' + str(layer_idx)] = W_new
            self._param_dict['b' + str(layer_idx)] = b_new

        return





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
        """
        # Initializing lists to hold loss record
        per_epoch_loss_train = []
        per_epoch_loss_val = []

        # Train (1 epoch = 1 full forward pass + 1 full backward pass)
        for i in range(self._epochs):
            print("Training epoch ", i+1, "...")

            # Shuffle the training data for each epoch of training
            shuffle_arr = np.concatenate([X_train, y_train], axis=1)
            np.random.shuffle(shuffle_arr)

            # Calculate how many dimensions y_train had
            y_train_dims = y_train.shape[1]
            shuffle_arr_dims = shuffle_arr.shape[1]
            cutoff_value = shuffle_arr_dims - y_train_dims

            # Recover X and y batches
            X_train = shuffle_arr[:, :cutoff_value]
            y_train = shuffle_arr[:, cutoff_value:]
            num_batches = np.ceil(X_train.shape[0]/self._batch_size)
            X_batch = np.array_split(X_train, num_batches)
            y_batch = np.array_split(y_train, num_batches)

            loss_history_train = []
            loss_history_val = []

            # Iterating through batches (full for loop is one epoch of training)
            for X_train, y_train in zip(X_batch, y_batch):

                # Forward pass
                output, cache = self.forward(X_train)

                print(cache)
                # Calculate training loss
                assert (self._loss_func in ["mse", "bce"]), "Loss function unrecognized"
                if self._loss_func == "mse":
                    loss_train = self._mean_squared_error(y_train, output)
                else: # self._loss_func == "bce"
                    loss_train = self._binary_cross_entropy(y_train, output)


                # Add current training loss to loss history record
                loss_history_train.append(loss_train)

                # Backward pass
                grad_dict = self.backprop(y_train, output, cache)

                # Update parameters
                self._update_params(grad_dict)

                # Validation pass
                output_val = self.predict(X_val)

                # Calculate validation loss
                assert (self._loss_func in ["mse", "bce"]), "Loss function unrecognized"
                if self._loss_func == "mse":
                    loss_val = self._mean_squared_error(y_val, output_val)
                else: # self._loss_func == "bce"
                    loss_val = self._binary_cross_entropy(y_val, output_val)
    
                # Add current validation loss to loss history record
                loss_history_val.append(loss_val)

            # Record average losses for the epoch
            per_epoch_loss_train.append(sum(loss_history_train)/len(loss_history_train))
            per_epoch_loss_val.append(sum(loss_history_val)/len(loss_history_val))

            print("Epoch ", i+1, " done!")

        print("Training done!")
        return per_epoch_loss_train, per_epoch_loss_val



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
        return y_hat




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
        exp_pred = np.exp(-1 * Z)
        nl_transform = 1/(1 + exp_pred)
        return nl_transform




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
        nl_transform = np.maximum(0, Z)
        return nl_transform




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
        dL_dA = dA
        dA_dZ = self._sigmoid(Z) * (1-self._sigmoid(Z))
        dL_dZ = dL_dA * dA_dZ

        return dL_dZ




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
        dL_dA = dA
        dA_dZ = Z
        dA_dZ[dA_dZ <= 0] = 0 # Technically, the derivative if Z==0 is undefined.
        dA_dZ[dA_dZ > 0] = 1
        dL_dZ = dL_dA * dA_dZ

        return dL_dZ




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
        epsilon = 1e-5   
        y_0 = (1-y) * np.log(1-y_hat + epsilon)      # Values where y=1 will be 0
        y_1 = y * np.log(y_hat + epsilon)            # Values where y=0 will be 0
        loss = -1 * np.mean(y_0 + y_1, axis=0)

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
        num_terms = y.shape[0]
        dL_dy_hat = (((1-y)/(1-y_hat)) - (y/y_hat)) / num_terms
        dA = dL_dy_hat

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
        loss = np.mean(np.power(y-y_hat, 2))
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
        num_terms = y.shape[0]
        dL_dy_hat = -2 * (y-y_hat) / num_terms
        dA = dL_dy_hat

        return dA

    def _loss_function(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Loss function, computes loss given y_hat and y. This function is
        here for the case where someone would want to write more loss
        functions than just binary cross entropy.
        Args:
            y: ArrayLike
                Ground truth output.
            y_hat: ArrayLike
                Predicted output.
        Returns:
            loss: float
                Average loss of mini-batch.
        """
        pass

    def _loss_function_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        This function performs the derivative of the loss function with respect
        to the loss itself.
        Args:
            y (array-like): Ground truth output.
            y_hat (array-like): Predicted output.
        Returns:
            dA (array-like): partial derivative of loss with respect
                to A matrix.
        """
        pass
