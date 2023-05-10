import pickle
import unittest
from utils_data import read_dataset

import Kmeans as km
from Kmeans import *
from utils import *


# unittest.TestLoader.sortTestMethodsUsing = None

class TestCases(unittest.TestCase):

    def setUp(self):
        train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, \
        test_color_labels = read_dataset(ROOT_FOLDER='./images/', gt_json='./images/gt.json')

    def test_01_retrieval_by_color(self):
        images = []


if __name__ == "__main__":
    unittest.main()
