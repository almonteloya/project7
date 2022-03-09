# BMI 203 Project 7: Neural Network

import numpy as np
from nn.nn import NeuralNetwork
import pytest 
from nn.preprocess import one_hot_encode_seqs, sample_seqs


# TODO: Write your test functions and associated docstrings below.


## First we write a very simple network
nn_ar = [{'input_dim': 2, 'output_dim':2, 'activation' : 'Relu'}, {'input_dim': 2, 'output_dim': 2, 'activation': 'Relu'}]


def test_forward():
    pass


def test_single_forward():
    pass


def test_single_backprop():
    pass


def test_predict():
    pass


def test_binary_cross_entropy():
    pass


def test_binary_cross_entropy_backprop():
    pass


def test_mean_squared_error():
    pass


def test_mean_squared_error_backprop():
    pass


def test_one_hot_encode():
    out_hot = one_hot_encode_seqs("AGA","CT")
    assert (out_hot == [[1, 0, 0, 0,0, 0, 0, 1, 1, 0, 0, 0],[0, 1, 0, 0, 0, 0, 1, 0]])


def test_sample_seqs():
    pass
