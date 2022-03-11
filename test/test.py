# BMI 203 Project 7: Neural Network

import numpy as np
from nn.nn import NeuralNetwork
import pytest 
from nn.preprocess import one_hot_encode_seqs, sample_seqs


"""
For most of the tests, I used a simple network, with dimension 2,2
When the network is first created random numbers are assigned to the weights and bias. 
For further simplicity this weights and bias were changed for simpler numbers
"""

## First we write a very simple network
nn_ar_relu = [{'input_dim': 2, 'output_dim':2, 'activation' : 'Relu'}, {'input_dim': 2, 'output_dim': 2, 'activation': 'Relu'}]
nn_ar_sig = [{'input_dim': 2, 'output_dim':2, 'activation' : 'Sigmoid'}, {'input_dim': 2, 'output_dim': 2, 'activation': 'Sigmoid'}]

lr = .1
seed = 42
batch_size = 50
epochs = 20
loss_function = "mse"
## Creationg an instance of the network 

testpy = NeuralNetwork(nn_ar_relu,lr, seed, batch_size,epochs,loss_function)
testpy_sig = NeuralNetwork(nn_ar_sig,lr, seed, batch_size,epochs,loss_function)


## Create a small random input 
X = np.array([[2],[1]])

## Make a simpler W1 array
simpleW = np.array([[.5],[-.5]])
# Update the network
testpy._param_dict['W1'] = simpleW

## Make a simpler b1 array
simpleb = np.array([[1],[.5]])
# Update the network
testpy._param_dict['b1'] = simpleb

def test_forward():

    """
    Forward: 

    During the forward method Z is calculated.
    The formula for Z = A_prev.dot(W_curr.T) + b_curr.T
    In the first layer A_prev == X. We also have the values for W1 and b1
    Use the formula and get the result using R
    """

    anwer_R = np.array([[ 2. , -0.5],
                        [ 1.5,  0. ]])
    A_curr, cache = testpy.forward(X)

    assert(np.all(cache['Z1'] == anwer_R))


def test_single_forward():

    """
    Single forward:

    We know now we have the correct result for Z.
    During single_forward an activation function is applied to calculate A.
    Let's try with Relu 
    For Relu we know if the value is less than 0 then 0 if not then use number from Z.
    So the answer was calculated manually
    """

    #True anawer

    answer_A= np.array([[2. , 0. ],
                        [1.5, 0. ]])
    A_, Z_ = testpy._single_forward(simpleW, simpleb, X, "Relu")

    assert(np.all(A_ == answer_A))

def test_predict():
    """
    Test 1: since we already proved forward is correct we then prove they give the same results 

    Test 2: For the sigmoid fuction the prediction should be between 1 and 0

    """

    y_pred, _ = testpy.predict(X)
    y_predf,_ = testpy.forward(X)

    assert(np.all(y_pred == y_predf))


    X= np.array([2,1]).reshape(1,2)
    y_pred_sigmoid, _testpy_sig.predict(X)

    assert( np.all((1 > y_pred_sigmoid) & (y_pred_sigmoid > 0)))


def test_single_backprop():

    """
    For testing the single_backprop we previosly define y, ypred, Z and A
    We used the _mean_squared_error_backprop
    """
    ## This are our pre-set values
    y_pred, _ = testpy.predict(X)
    y = np.array([[0.5], [-0.5]]) 
    Z1= np.array([[ 2. , -0.5],
                  [ 1.5,  0. ]])
    A1 = np.array([[2. , 0. ],
                   [1.5, 0. ]])
    da = testpy._mean_squared_error_backprop(y,y_pred)

    ## We calculate the derivate of relu with the following rule
    # if z == 0 then 1 (because is not defined)
    # if z < 0 then 0
    # if z > 0 then z
    # Following this logic then:
    derivateZ = np.array([[ 2. , 0],[ 1.5,  1]])
    manual_da_prev = np(da*derivateZ).dot(simpleW)


    dA_prev, dW_curr, db_curr = testpy._single_backprop(
    W_curr= simpleW,
    b_curr= simpleb,
    Z_curr = Z1,
    A_prev=  A1,
    dA_curr= testpy._mean_squared_error_backprop(y,y_pred),
    activation_curr= 'Relu')


    assert(dA_prev == manual_da_prev)



def test_binary_cross_entropy():
    """
    Based on binary cross entropy formula https://cdn.analyticsvidhya.com/wp-content/uploads/2021/03/Screenshot-from-2021-03-03-11-33-29.png
    1/N * -sum(y*log(p) + 1-y * log(1-p))
    We define y and y_hat and plug the formula using R
    """
    y = np.array([.5,.4,0])
    y_hat = np.array([.5,.3,.2])

    R_anwer =  0.5372949
    bce = testpy_sig._binary_cross_entropy(y,y_hat)
    ## Since we have a difference of decimals we'll round them
    R_anwer = np.around(R_anwer, decimals=5) 
    bce = np.around(bce, decimals=5) 
    
    assert(R_anwer == bce)



def test_binary_cross_entropy_backprop():
    """
    This is the formula for the derivate for  the binary cross entropy 
    ((1-y)/(1-y_hat) - (y/y_hat)) / len(y)
    We define y and y hat then we plug the formula using R
    """
    y = np.array([[0.5],[0.4], [0]])
    y_hat = np.array([[.5],[.3],[.2]])

    answe_R = np.array([[0.0000000] ,[-0.1587302], [0.4166667]])
    bceb = testpy._binary_cross_entropy_backprop(y,y_hat)

    answe_R = np.around(answe_R, decimals=5) 
    bceb = np.around(bceb, decimals=5)

    assert(answe_R == bceb)




def test_mean_squared_error():
    """
    The formula for MSE is sum(y-y_pred)**2 / n 
    We define y and y_pred, then we compare with the R result
    """
    y = np.array([1.2,3.5,4])
    y_hat = np.array([1,3,5])

    answer_R = 0.43

    mse_py = testpy_sig._mean_squared_error(y,y_hat)

    assert (answer_R == mse_py)


def test_mean_squared_error_backprop():

    """
    The formula for the derivate is -(2*(y - y_pred)/len(y))
    Since we know the y_pred and determined the y values 
    We can simply plug the formula on R
    """
    y_pred, _ = testpy.predict(X)
    y = np.array([[0.5],
       [-0.5]])
    ## This is the R result from te formula

    mse_R = np.array([[-0.3788796],[0.6219598]])
    mse_py = testpy._mean_squared_error_backprop(y,y_pred)

    ## Since we have a difference of decimals we'll round them
    mse_R_r = np.around(mse_R, decimals=7) 
    mse_P_r = np.around(mse_py, decimals=7) 
    
    assert(np.all(mse_R_r - mse_P_r == 0))


def test_one_hot_encode():
    """
    The correct answer was calculated manually based on the example 
    """
    out_hot = one_hot_encode_seqs(["AGA","CT"])
    assert (out_hot == [[1, 0, 0, 0,0, 0, 0, 1, 1, 0, 0, 0],[0, 1, 0, 0, 0, 0, 1, 0]])


def test_sample_seqs():
    """
    Using sample_seqs, returns a 3:1 ratio of positive and negative sequence
    The test is to make sure this ratio is correct 
    """
    rap1_postive = read_text_file("data/rap1-lieb-positives.txt")
    rap1_negative = read_text_file("data/yeast-upstream-1k-negative.fa")

    sampled_seqs,sampled_labels = sample_seqs(rap1_postive,rap1_negative)

    assert(sum(sampled_labels==1)*3 == sum(sampled_labels==0))


