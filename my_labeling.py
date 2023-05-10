__authors__ = ['1630568', '1636442']
__group__ = 'DM.12'

from utils_data import read_dataset

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
    
