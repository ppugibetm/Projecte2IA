import pickle
import unittest
from utils_data import read_dataset

from Kmeans import *
from utils import *
from my_labeling import *
import numpy as np

# unittest.TestLoader.sortTestMethodsUsing = None

class TestCases(unittest.TestCase):
    
    def setUp(self):
        self.train_imgs, self.train_class_labels, self.train_color_labels, self.test_imgs, self.test_class_labels, \
        test_color_labels = read_dataset(ROOT_FOLDER='./images/', gt_json='./images/gt.json')
        
        classes = list(set(list(self.train_class_labels) + list(self.test_class_labels)))

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
        
        np.testing.assert_array_equal(retrieval_combined(images, shapes, colors, 'Shorts', 'Orange'), [0, 9])
        
        
    def test_04_Kmeans_statistics(self):
        np.random.seed(123)
        with open('./test/test_cases_kmeans.pkl', 'rb') as f:
            self.test_cases = pickle.load(f)
            
        km = []
            
        for ix, input in enumerate(self.test_cases['input']):
            km.append(KMeans(input, self.test_cases['K'][ix]))
            
        Kmean_statistics(km, 5)
        
    
    def test_05_get_shape_accuracy(self):
        shapes = self.train_class_labels[:10]
        
        error = ['Shorts', 'Heels', 'Shorts', 'Shirts', 'Jeans', 'Jeans', 'Socks', 'Shirts', 'Sandals', 'Shorts']
        
        np.testing.assert_array_equal(get_shape_accuracy(error, shapes), 90.0)
        
        
    def test_06_get_color_accuracy(self):
        colors = self.train_color_labels[:10]
        
        error = np.array([list(['Black', 'error', 'White']),       list(['Brown', 'error', 'Orange', 'White']),       list(['White', 'error']),       list(['Black', 'Brown', 'Orange', 'White']),       list(['Grey', 'Red', 'White']), list(['Blue', 'Pink', 'White']),       list(['Blue', 'White']), list(['Blue', 'Orange', 'White']),       list(['Black', 'Grey', 'White']),       list(['Brown', 'Grey', 'error', 'White'])], dtype=object)
        
        np.testing.assert_array_equal(get_color_accuracy(colors, error), 87.1)
         

if __name__ == "__main__":
    unittest.main()
