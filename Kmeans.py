__authors__ = ['1630568', '1636442']
__group__ = '2_5'

import numpy as np
import utils


class KMeans:

    def __init__(self, X, K=1, options=None):
        """
         Constructor of KMeans class
             Args:
                 K (int): Number of cluster
                 options (dict): dictionary with options
            """
        self.num_iter = 0
        self.K = K
        self._init_X(X)
        self._init_options(options)  # DICT options
        self.WCD = None
        self.labels = None
        self.centroids = None
        self.old_centroids = None

    #############################################################
    ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
    #############################################################

    def _init_X(self, X):
        """Initialization of all pixels, sets X as an array of data in vector form (PxD)
            Args:
                X (list or np.array): list(matrix) of all pixel values
                    if matrix has more than 2 dimensions, the dimensionality of the sample space is the length of
                    the last dimension
        """

        array_x = np.array(X)

        if array_x.dtype != np.float64:
            array_x = array_x.astype(np.float64)

        if array_x.ndim > 2:
            shape = array_x.shape
            array_x = np.reshape(array_x, (shape[0] * shape[1], shape[2]))

        self.X = array_x

    def _init_options(self, options=None):
        """
        Initialization of options in case some fields are left undefined
        Args:
            options (dict): dictionary with options
        """
        if options is None:
            options = {}
        if 'km_init' not in options:
            options['km_init'] = 'first'
        if 'verbose' not in options:
            options['verbose'] = False
        if 'tolerance' not in options:
            options['tolerance'] = 0
        if 'max_iter' not in options:
            options['max_iter'] = np.inf
        if 'fitting' not in options:
            options['fitting'] = 'WCD'  # within class distance.

        # If your methods need any other parameter you can add it to the options dictionary
        self.options = options

        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################

    def _init_centroids(self):
        """
        Initialization of centroids
        """
        self.centroids = []
        n = 1

        match self.options['km_init']:
            case 'random':
                while n < self.K:
                    x = np.random.randrange(self.K - 1)
                    if self.add_centroids(x):
                        n += 1
            case 'first' | _:
                self.centroids.append(self.X[0])
                aux = 0
                while n < self.K:
                    aux += 1
                    if self.add_centroids(aux):
                        n += 1

    def add_centroids(self, x):
        already = False

        for centroid in self.centroids:
            if np.array_equal(self.X[x], centroid):
                already = True

        if not already:  # == False:
            self.centroids.append(self.X[x])
            return True
        return False

    def get_labels(self):
        """        Calculates the closest centroid of all points in X
        and assigns each point to the closest centroid
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        array = []

        for i in self.X:
            min_dist = 100000000000
            n = -1
            for x, j in enumerate(self.centroids):
                dist = np.sqrt(pow(i[0] - j[0], 2) + pow(i[1] - j[1], 2) + pow(i[2] - j[2], 2))
                if dist < min_dist:
                    min_dist = dist
                    n = x
            array.append(n)

        self.labels = array

    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        self.old_centroids = self.centroids
        new_centroids = [[] for i in range(self.K)]

        for i in range(len(self.X)):
            new_centroids[self.labels[i]].append(self.X[i])

        for i in range(len(new_centroids)):
            new_centroids[i] = np.average(np.array(new_centroids[i]), 0)

        self.centroids = new_centroids

    def converges(self):
        return np.allclose(self.centroids, self.old_centroids, atol=self.options['tolerance'])

    def fit(self):
        self._init_centroids()
        max_iter = self.options['max_iter']
        trobat = False
        while (self.num_iter <max_iter) and (trobat == False):
            self.get_labels()
            self.get_centroids()
            if self.converges() == True:
                trobat = True
            else:
                self.num_iter += 1

    def withinClassDistance(self):
        dist = 0
        for i in range(len(self.centroids)):
            pixels_per_centroid = np.where(self.labels == i)[0]
            dist += np.sum((self.X[pixels_per_centroid] - self.centroids[i]) ** 2)
        return dist / len(self.X)

    def find_bestK(self, max_K):
        self.K = 2
        self.fit()
        anterior_WCD = self.withinClassDistance()
        for i in range(3, max_K):
            self.K = i
            self.fit()
            WCD = self.withinClassDistance()
            dec = (100 - (100 * (WCD / anterior_WCD)))
            anterior_WCD = WCD
            if dec < 20:
                self.K = i - 1
                self.fit()
                break

def distance(X, C):
    """
    Calculates the distance between each pixel and each centroid
    Args:
        X (numpy array): PxD 1st set of data points (usually data points)
        C (numpy array): KxD 2nd set of data points (usually cluster centroids points)
    Returns:
        dist: PxK numpy array position ij is the distance between the
        i-th point of the first set an the j-th point of the second set
    """
    dist = []

    for i in X:
        aux_dist = []
        for j in C:
            aux_dist.append(np.sqrt(pow(i[0] - j[0], 2) + pow(i[1] - j[1], 2) + pow(i[2] - j[2], 2)))
        dist.append(aux_dist)

    dist_ = np.array(dist)
    shapeX = X.shape
    shapeC = C.shape

    dist_ = np.reshape(dist_, (shapeX[0], shapeC[0]))

    return dist_


def get_colors(centroids):
    """
    for each row of the numpy matrix 'centroids' returns the color label following the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroid points)
    Returns:
        labels: list of K labels corresponding to one of the 11 basic colors
    """

    #########################################################
    ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    ##  AND CHANGE FOR YOUR OWN CODE
    #########################################################
    return list(utils.colors)
