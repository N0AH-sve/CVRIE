##
## EPITECH PROJECT, 2026
## G-AIA-400-NAN-4-1-cvrie-2
## File description:
## preprocessing
##

import matplotlib.image as plt_img
import numpy as np

def image_to_array(image):
    image = plt_img.imread(image)
    # print(image.shape)
    return image

def do_gray_scale(image):
    if image.ndim == 3:
        return image[:, :, 0]
    return image

def interpolation(image, sizeX, sizeY):
    # à implémenter, interpolation bilinéaire
    # pour resize les images sans les déformer
    # return les images dans un format 124 x 124
    return

def resize_image(image, sizeX, sizeY):
    print("Resizing image to: ", sizeX, sizeY)
    return np.resize(image, (sizeX, sizeY))

def preprocess_image(image, size):
    image = image_to_array(image)
    image = do_gray_scale(image)
    image = resize_image(image, size[0], size[1])
    # possibilité d'ajouter des steps de preprocessing supplémentaires (normalisation, etc.)
    print(image.shape)
    return image

def flatten_image_array(matrix_array):
    n_samples = matrix_array.shape[0]
    flat_matrix_array =  matrix_array.reshape(n_samples, -1)
    return flat_matrix_array

def test():
    return "preprocessing"
