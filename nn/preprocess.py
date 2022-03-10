# BMI 203 Project 7: Neural Network


# Importing Dependencies
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike
import random



# Defining preprocessing functions
def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one hot encoding of a list of nucleic acid sequences
    for use as input into a fully connected neural net.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence
            length due to the one hot encoding scheme for nucleic acids.

            For example, if we encode 
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
    """
    sus_mat = {"A": [1, 0, 0, 0], "T": [0, 1, 0, 0],"C": [0, 0, 1, 0],"G":[0, 0, 0, 1]}
    ## DICTIONARY with sustitution values

    ## Function to split char in a str
    def _split(word):
        return [char for char in word]

    one_list=[]
    for st in seq_arr:
        split_str = _split(st)
        one_code=[]
        for char in split_str:
            one_code = one_code + sus_mat[char]
        one_list.append(one_code)
    return one_list


def sample_seqs(positive,negative):
    """
    This function should sample your sequences to account for class imbalance. 
    Consider this as a sampling scheme with replacement.
    
    Args:
        positive: List[str]
            List of all positive sequences.
        negative: List[str]
            List of negative sequences

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """
    ## Remove the headers from the negative sequeces
    negative_filter = [x for x in negative if not x.startswith('>')]
    ## get a ratio 2:1
    sub_negative = random.sample(negative_filter, len(positive)*3)
    ## Then get the first 17 nucleotides for the negative 
    sub_negative = [i[:17] for i in sub_negative]
    ## Now we can start put everything togueter
    sampled_seqs = positive + sub_negative
    x = np.array([1, 0]) ## Create 
    sampled_labels = np.repeat(x, [len(positive), len(sub_negative)], axis=0)
    
    return sampled_seqs, sampled_labels
