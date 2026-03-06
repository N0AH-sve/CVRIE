##
## EPITECH PROJECT, 2026
## G-AIA-400-NAN-4-1-cvrie-2
## File description:
## preprocessing
##

import matplotlib.pyplot as plt
import matplotlib.image as imread
import numpy as np


def image_to_array(image):
    image = imread.imread(image)
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

def test():
    return "preprocessing"
