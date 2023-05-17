__authors__ = ['1630568', '1636442']
__group__ = 'DM.12'

from utils_data import read_dataset
import time
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, \
        test_color_labels = read_dataset(ROOT_FOLDER='./images/', gt_json='./images/gt.json')

    # List with all the existent classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    # You can start coding your functions here

def retrieval_by_color(images, labels, question):
    matches = []
    
    for image in range(len(images)):
        color_tag = labels[image]
        if any(color == question for color in color_tag):
            matches.append(image)
            
    return matches

def retrieval_by_shape(images, labels, question):
    matches = []
    
    for image in range(len(images)):
        shape_tag = labels[image]
        if shape_tag == question:
            matches.append(image)
            
    return matches

def retrieval_combined(images, labels_shape, labels_color, question_shape, question_color):
    matches = []
    
    matches_color = retrieval_by_color(images, labels_color, question_color)
    
    for image in matches_color:
        shape_tag = labels_shape[image]
        if shape_tag == question_shape:
            matches.append(image)
            
    return matches


def Kmean_statistics(kmeans_list, Kmax):
    wcd_lists = [[] for i in range(len(kmeans_list))]
    time_lists = [[] for i in range(len(kmeans_list))]
    n_iters_list = list(range(2, Kmax + 1))
    
    for i, kmeans in enumerate(kmeans_list):
        wcd_list = []
        time_list = []
        
        for k in range(2, Kmax + 1):
            kmeans.K = k
            start = time.time()
            kmeans.fit()
            end = time.time()
            
            wcd_list.append(kmeans.WCD)
            time_list.append(end - start)
        
        wcd_lists[i] = wcd_list
        time_lists[i] = time_list
    
    plt.figure("Kmean statistics")
    
    plt.subplot(2, 1, 1)
    for i, wcd_list in enumerate(wcd_lists):
        plt.plot(n_iters_list, wcd_list, label=f'KMeans {i+1}')
    
    plt.xlabel('K')
    plt.ylabel('WCD')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    for kmeans_index, time_list in enumerate(time_lists):
        plt.plot(n_iters_list, time_list, label=f'KMeans {kmeans_index+1}')
    
    plt.xlabel('K')
    plt.ylabel('Computation Time (s)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    

def get_shape_accuracy(labels, gt):
    errors = 0
    for i, label in enumerate(labels):
        if label != gt[i]:
            errors += 1
    
    return (1 - (errors/len(labels))) * 100


def get_color_accuracy(labels, gt):
    errors = 0
    total_len = 0
    for i, label in enumerate(labels):
        for x in label:
            total_len += 1
            if x not in gt[i]:
                errors += 1
    per = round((1 - (errors/total_len)) * 100, 2)
    return per

