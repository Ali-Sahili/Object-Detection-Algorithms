import os
import json
import argparse
from utils.helpers import label_map, parse_annotation


parser = argparse.ArgumentParser(description='Create lists from VOC dataset')
parser.add_argument('--train_path', default='./datasets/VOC2007_train/VOCdevkit/VOC2007/', 
                                    help='Dataset directory')
parser.add_argument('--test_path', default='./datasets/VOC2007_test/VOCdevkit/VOC2007/', 
                                   help='Dataset directory')

def create_data_lists(voc07_path_train, voc07_path_test, output_folder):
    """
    Create lists of images, bounding boxes and labels of the objects in these images, and 
    save these lists to a json file.

    :param voc07_path: path to the 'VOC2007' folder
    :param voc12_path: path to the 'VOC2012' folder
    :param output_folder: folder where the JSONs must be saved
    """
    voc07_path_train = os.path.abspath(voc07_path_train)
    voc07_path_test = os.path.abspath(voc07_path_test)
    
    train_images = list()
    train_objects = list()
    n_objects = 0

    # Training data
    for path in [voc07_path_train]:

        # Find IDs of images in training data
        with open(os.path.join(path, 'ImageSets/Main/trainval.txt')) as f:
            ids = f.read().splitlines()

        for id in ids:
            # Parse annotation's XML file
            objects = parse_annotation(os.path.join(path, 'Annotations', id + '.xml'))
            if len(objects['boxes']) == 0:
                continue
            n_objects += len(objects)
            train_objects.append(objects)
            train_images.append(os.path.join(path, 'JPEGImages', id + '.jpg'))

    assert len(train_objects) == len(train_images)

    # Save to file
    with open(os.path.join(output_folder, 'TRAIN_images.json'), 'w') as j:
        json.dump(train_images, j)
    with open(os.path.join(output_folder, 'TRAIN_objects.json'), 'w') as j:
        json.dump(train_objects, j)
    with open(os.path.join(output_folder, 'label_map.json'), 'w') as j:
        json.dump(label_map, j)  # save label map too

    print('\nThere are %d training images containing a total of %d objects. Files have been saved to %s.' % (len(train_images), n_objects, os.path.abspath(output_folder)))

    # Test data
    test_images = list()
    test_objects = list()
    n_objects = 0

    # Find IDs of images in the test data
    with open(os.path.join(voc07_path_test, 'ImageSets/Main/test.txt')) as f:
        ids = f.read().splitlines()

    for id in ids:
        # Parse annotation's XML file
        objects = parse_annotation(os.path.join(voc07_path_test, 'Annotations', id + '.xml'))
        if len(objects) == 0:
            continue
        test_objects.append(objects)
        n_objects += len(objects)
        test_images.append(os.path.join(voc07_path_test, 'JPEGImages', id + '.jpg'))

    assert len(test_objects) == len(test_images)

    # Save to file
    with open(os.path.join(output_folder, 'TEST_images.json'), 'w') as j:
        json.dump(test_images, j)
    with open(os.path.join(output_folder, 'TEST_objects.json'), 'w') as j:
        json.dump(test_objects, j)

    print('\nThere are %d test images containing a total of %d objects. Files have been saved to %s.' % (len(test_images), n_objects, os.path.abspath(output_folder)))


if __name__ == '__main__':
    args = parser.parse_args()
    create_data_lists(voc07_path_train=args.train_path, voc07_path_test=args.test_path,
                      output_folder='datasets/')
