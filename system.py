"""
Solution outline for the COM2004/3004 assignment.

This solution will run K Nearest Neighbour classificaiton with Principle Component Analysis 
to perform dimentionality reducition down to 10 dimentions.
Work done by Ethan Barker

"""
from typing import List

import numpy as np
import scipy.linalg

N_DIMENSIONS = 10


def classify(train: np.ndarray, train_labels: np.ndarray, test: np.ndarray) -> List[str]:
   
    """Classify a set of feature vectors using a training set.

    My implemtnation uses KNN classification wich classifies the current data point 
    with how its neighbours are classified. This is done via using the cosine distance
    in the method I have used.

    Args:
        train (np.ndarray): 2-D array storing the training feature vectors.
        train_labels (np.ndarray): 1-D array storing the training labels.
        test (np.ndarray): 2-D array storing the test feature vectors.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """

    #Use NN Classifier taked from lab 6 but alterations have been made by myself
    x = np.dot(test, train.transpose())
    modtest = np.sqrt(np.sum(test * test, axis = 1))
    modtrain = np.sqrt(np.sum(train * train, axis = 1))
    distance = x / np.outer(modtest, modtrain.transpose()) #Use the cosine distance
    
    k = 1
    if k == 1:
        nearest = np.argmax(distance, axis = 1)
        label = train_labels[nearest]
        return label
    else:
        # k-Nearest neighbour classification
        knearest = np.argsort(-distance, axis=1)[:, :k]
        klabel = train_labels[knearest]
        return np.array(klabel)



# The functions below must all be provided in your solution. Think of them
# as an API that it used by the train.py and evaluate.py programs.
# If you don't provide them, then the train.py and evaluate.py programs will not run.
#
# The contents of these functions are up to you but their signatures (i.e., their names,
# list of parameters and return types) must not be changed. The trivial implementations
# below are provided as examples and will produce a result, but the score will be low.


def reduce_dimensions(data: np.ndarray, model: dict) -> np.ndarray:
    """Reduce the dimensionality of a set of feature vectors down to N_DIMENSIONS.

    The feature vectors are stored in the rows of 2-D array data, (i.e., a data matrix).
    The dummy implementation below simply returns the first N_DIMENSIONS columns.

    Args:
        data (np.ndarray): The feature vectors to reduce.
        model (dict): A dictionary storing the model data that may be needed.

    Returns:
        np.ndarray: The reduced feature vectors.
    """
    #Using PCA for the A martix to tranform 40 dimentions
    v = np.array(model["eigenvector"])
    #Find the mean and subtract it from the datapoints to find the centre
    mean = np.mean(data)
    center = data - mean
    #Use this center on the PCA axis
    pca_train_data = np.dot(center, v)
    
    return pca_train_data


def process_training_data(fvectors_train: np.ndarray, labels_train: np.ndarray) -> dict:
    """Process the labeled training data and return model parameters stored in a dictionary.

    Here PCA is performed to reduce the noise from the feature vectors and to reduce 
    the feature dimention down to N dimentions which is 10. It then stores the eignevectors 
    of the training data. 

    Args:
        fvectors_train (np.ndarray): training data feature vectors stored as rows.
        labels_train (np.ndarray): the labels corresponding to the feature vectors.

    Returns:
        dict: a dictionary storing the model data.
    """

    #Performing PCA analysis taken from Lab 7 with edits made by myself
    covx = np.cov(fvectors_train, rowvar = 0)
    N = covx.shape[0]
    w, v = scipy.linalg.eigh(covx, eigvals = (N - N_DIMENSIONS , N -1)) #Returns the eigenvectors (PCA axis)
    v = np.flip(v, 1)
    #Creates the dictionary storing model data
    model = {}
    model["eigenvector"]= v.tolist()
    model["labels_train"] = labels_train.tolist()
    fvectors_train_reduced = reduce_dimensions(fvectors_train, model)
    model["fvectors_train"] = fvectors_train_reduced.tolist()
    return model


def images_to_feature_vectors(images: List[np.ndarray]) -> np.ndarray:
    """Takes a list of images (of squares) and returns a 2-D feature vector array.

    In the feature vector array, each row corresponds to an image in the input list.

    Args:
        images (list[np.ndarray]): A list of input images to convert to feature vectors.

    Returns:
        np.ndarray: An 2-D array in which the rows represent feature vectors.
    """
    h, w = images[0].shape
    n_features = h * w
    fvectors = np.empty((len(images), n_features))
    for i, image in enumerate(images):
        fvectors[i, :] = image.reshape(1, n_features)

    return fvectors


def classify_squares(fvectors_test: np.ndarray, model: dict) -> List[str]:
    """Run classifier on a array of image feature vectors presented in an arbitrary order.

    Note, the feature vectors stored in the rows of fvectors_test represent squares
    to be classified. The ordering of the feature vectors is arbitrary, i.e., no information
    about the position of the squares within the board is available.

    Args:
        fvectors_test (np.ndarray): An array in which feature vectors are stored as rows.
        model (dict): A dictionary storing the model data.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """

    # Get some data out of the model. It's up to you what you've stored in here
    fvectors_train = np.array(model["fvectors_train"])
    labels_train = np.array(model["labels_train"])
    """
    no_pawns_data = fvectors_train[labels_train != ("p" or "P"), :]
    no_pawns_label = np.delete(labels_train, np.where(labels_train == ("p" or "P"))
    """
    
    # Call the classify function.
    labels = classify(fvectors_train, labels_train, fvectors_test)

    return labels


def classify_boards(fvectors_test: np.ndarray, model: dict) -> List[str]:
    """
    #Split the boards into seperate boards containing 64 squares
    boards = [data[x:x+64] for x in range(0,len(data), 64)]
    #Num represents the board number
    num = 0
    for board in boards:
        #In the first row a value can not be assigned as p or P
        for i, v in enumerate(board[:8]):
            if v == ("p" or "P"):
                 board[i] = classify(no_pawns_data,no_pawns_label,fvectors_test[num*64 + i, :])
        #In the final row a value can not be assigned as p or P
        for i, v in enumerate(board[56:]):
            if v == ("p" or "P"):
                 board[i] = classify(no_pawns_data,no_pawns_label,fvectors_test[num*64 + i + 56, :])
                               
    """
    
    return classify_squares(fvectors_test, model)