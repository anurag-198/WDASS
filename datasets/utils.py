import os

import tqdm
import cv2
from yaml import load, dump, FullLoader
import numpy as np


try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper, FullLoader



def points_data_to_yaml(data_file_path, output_file_path="points_data.yaml"):
    """
        Convert point data file to yaml file
        Args:
            data_file_path: path to point data file
            output_file_path: path to output yaml file
    """
    if not os.path.exists(data_file_path):
        raise FileNotFoundError(f'File {data_file_path} not found')
    
    if not output_file_path.endswith('.yaml'):
        print(f'Output file path {output_file_path} must have .yaml extension')
        print('Adding .yaml extension to output file path')
        if '.' in output_file_path:
            output_file_path = output_file_path.split('.')[0] + '.yaml'
        else:
            output_file_path += '.yaml'

    
    co = np.load(data_file_path)

    
    images = []
    for i in range(len(co)) : 
        if ("cityscapes" in co[i][1]) :
            sp = co[i][1]
            sp = sp.replace("gtCoarse", "gtPoint2")
            images.append(str(sp))
    
    points_data =[]
    print("Converting to yaml file...")
    with open(output_file_path, "w") as f:
        for i in tqdm.tqdm(range(len(images))):
            if not os.path.exists(images[i]):
                raise FileNotFoundError(f'File {images[i]} not found')
            img = cv2.imread(images[i])
            rows_indices, col_indices = np.where(np.all(img != 255, axis=-1))
            coordinates = [(int(x), int(y)) for x, y in zip(rows_indices, col_indices)]
            _points = []
            for (x, y) in coordinates:
                _points.append({'x': x, 'y':y, 'c': int(img[x, y][0])})
            points_data.append({"image": images[i].split("/")[-1], "size": img.shape, "points": _points})
        print("\tDumping to yaml file...")
        dump(points_data, f)
        print("Done!")


def generate_images_from_yaml(yaml_file_path, output_dir="gtPoint2_training"):
    """
        Load points data from yaml file to generate images with points
        Args:
            yaml_file_path: path to yaml file
            output_dir: path to output directory
    """
    if not os.path.exists(yaml_file_path):
        raise FileNotFoundError(f'File {yaml_file_path} not found')
    
    if output_dir is None:
        print('Output directory not provided')
        print('Using yaml file directory as output directory')
        output_dir = os.path.dirname(yaml_file_path)
        # TODO check if the appended string "_trainng" is correct
        output_dir = os.path.join(output_dir, 'gtPoint2_training')

    if not os.path.exists(output_dir):
        print(f'Output directory {output_dir} does not exist')
        print('Creating output directory')
        os.makedirs(output_dir)

    with open(yaml_file_path) as yaml_file:
        points = load(yaml_file, Loader=FullLoader)

    print("Generating images with points...")

    for i in tqdm.tqdm(range(len(points))):
        # create an image with
        image_data = points[i]
        img = np.ones(image_data['size'][:2], np.uint8) * 255
        for point in image_data['points']:
            x = point['x']
            y = point['y']
            c = point['c']
            img[x, y] = c
        cv2.imwrite(os.path.join(output_dir, image_data['image']), img)
    print("Done!")


def make_dataset_folder(folder):
    """
    Create Filename list for images in the provided path

    input: path to directory with *only* images files
    returns: items list with None filled for mask path
    """
    items = os.listdir(folder)
    items = [(os.path.join(folder, f), '') for f in items]
    items = sorted(items)

    print(f'Found {len(items)} folder imgs')

    """
    orig_len = len(items)
    rem = orig_len % 8
    if rem != 0:
        items = items[:-rem]

    msg = 'Found {} folder imgs but altered to {} to be modulo-8'
    msg = msg.format(orig_len, len(items))
    print(msg)
    """

    return items

if __name__=="__main__":
    generate_images_from_yaml("file_2975.yaml", "gtPoint2_training")