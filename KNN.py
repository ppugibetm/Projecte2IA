__authors__ = ['1630568', '1636442']
__group__ = 'DM.12'

import numpy as np
import math
import operator
from scipy.spatial.distance import cdist


class KNN:
    def __init__(self, train_data, labels):
        self._init_train(train_data)
        self.labels = np.array(labels)
        #self.neighbours = np
        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################

    def _init_train(self, train_data):
        """
        initializes the train data
        :param train_data: PxMxNx3 matrix corresponding to P color images
        :return: assigns the train set to the matrix self.train_data shaped as PxD (P points in a D dimensional space)
        """
        array_train = np.array(train_data)

        if array_train.dtype != np.float64:
            array_train = array_train.astype(np.float64)

        if array_train.ndim > 2:
            shape = array_train.shape
            array_train = np.reshape(array_train, (shape[0], shape[1]*shape[2]))

        self.train_data = array_train

    def get_k_neighbours(self, test_data, k):
        """
        given a test_data matrix calculates de k nearest neighbours at each point (row) of test_data on self.neighbors
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the matrix self.neighbors is created (NxK)
                 the ij-th entry is the j-th nearest train point to the i-th test point
        """
        array_train = np.array(test_data)

        if array_train.dtype != np.float64:
            array_train = array_train.astype(np.float64)

        if array_train.ndim > 2:
            shape = array_train.shape
            array_train = np.reshape(array_train, (shape[0], shape[1]*shape[2]))

        test_data = array_train
        
        dist = cdist(test_data, self.train_data)
        self.neighbours = []
           
        for i in range(shape[0]):
            aux_dist = sorted(dist[i])
            indexes = []     
            for j in range(k):
                indexes.append(self.labels[np.where(dist[i] == aux_dist[j])[0]])
            self.neighbours.append(indexes)

        self.neighbours = np.array(self.neighbours)
        self.neighbours = np.reshape(self.neighbours, (shape[0], k))


    def get_class(self):
        """
        Get the class by maximum voting
        :return: 2 numpy array of Nx1 elements.
                1st array For each of the rows in self.neighbors gets the most voted value
                            (i.e. the class at which that row belongs)
                2nd array For each of the rows in self.neighbors gets the % of votes for the winning class
        """
        total = {}
        most_voted_for_row = []
        
        shape = self.neighbours.shape
        
        for i in range(shape[0]):
            classes = {}
            for j in range(len(self.neighbours[i])):
                aux = self.neighbours[i][j]
                if (aux in classes):
                    classes[aux] += 1
                    total[aux] += 1
                else:
                    classes[aux] = 1
                    total[aux] = 1
                    
            max_value = max(classes.values())
            
            for n, j in enumerate(classes):
                if (classes[j] == max_value):
                    keys_list = list(classes.keys())
                    most_voted_for_row.append(keys_list[n])
                    break
            
        return most_voted_for_row


    def predict(self, test_data, k):
        """
        predicts the class at which each element in test_data belongs to
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the output form get_class (2 Nx1 vector, 1st the class 2nd the  % of votes it got
        """

        self.get_k_neighbours(test_data, k)
        return self.get_class()
