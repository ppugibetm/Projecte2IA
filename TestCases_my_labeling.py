import pickle
import unittest
from utils_data import read_dataset

import Kmeans as km
from Kmeans import *
from utils import *
from my_labeling import *

# unittest.TestLoader.sortTestMethodsUsing = None

class TestCases(unittest.TestCase):
    
    

    def setUp(self):
        self.train_imgs, self.train_class_labels, self.train_color_labels, self.test_imgs, self.test_class_labels, \
        test_color_labels = read_dataset(ROOT_FOLDER='./images/', gt_json='./images/gt.json')

    def test_01_retrieval_by_color(self):
        images = self.train_imgs[:10]
        colors = self.train_color_labels[:10]
        
        questions_color = {'Black':[0, 3, 8], 'Orange':[0, 1, 3, 7, 9], 'White':[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'Brown':[1, 3, 9], 'Grey':[1, 4, 8, 9], 'Yellow':[2], 'Red':[4], 'Blue':[5, 6, 7], 'Pink':[5]}
        
        keys_questions_color = questions_color.keys()
        
        for question in keys_questions_color:
            np.testing.assert_array_equal(retrieval_by_color(images, colors, question), questions_color[question])
            
            
    def test_02_retrieval_by_shape(self):
        images = self.train_imgs[:10]
        shapes = self.train_class_labels[:10]
        
        questions_shape = {'Shorts':[0,2,9], 'Heels':[1], 'Shirts':[3, 7], 'Socks':[4, 6], 'Jeans':[5], 'Sandals':[8]}
       
        keys_questions_shape = questions_shape.keys()
        
        for question in keys_questions_shape:
            np.testing.assert_array_equal(retrieval_by_shape(images, shapes, question), questions_shape[question])
            
            
    def test_03_retrieval_combined(self):
        images = self.train_imgs[:10]
        shapes = self.train_class_labels[:10]
        colors = self.train_color_labels[:10]
        
        questions_color = {'Black':[0, 3, 8], 'Orange':[0, 1, 3, 7, 9], 'White':[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'Brown':[1, 3, 9], 'Grey':[1, 4, 8, 9], 'Yellow':[2], 'Red':[4], 'Blue':[5, 6, 7], 'Pink':[5]}
        questions_shape = {'Shorts':[0,2,9], 'Heels':[1], 'Shirts':[3, 7], 'Socks':[4, 6], 'Jeans':[5], 'Sandals':[8]}
       
        keys_questions_color = questions_color.keys()
        keys_questions_shape = questions_shape.keys()
        
        np.testing.assert_array_equal(retrieval_combined(images, shapes, colors, 'Shorts', 'Orange'), [0, 9])
        

if __name__ == "__main__":
    unittest.main()
