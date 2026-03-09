##
## EPITECH PROJECT, 2026
## G-AIA-400-NAN-4-1-cvrie-2
## File description:
## preprocessing
##

import matplotlib.image as plt_img
import numpy as np


def image_to_array(image_path):
    """
        Transform an image into a numpy array.
        Args:
            image_path (string): Path to the image to transform.
        Returns:
            array: Matrice of points of the image.
    """
    numpy_array = plt_img.imread(image_path).astype(np.float32)
    return numpy_array


def divide_gray_scale(image):
    """    Divide the image into its gray scale values.
    Args:
        image (numpy array): Matrice of points of the image.
    Returns:
        array: Matrice of points of the image in gray scale.
    """
    if image.ndim == 3:
        return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.float32)
    return image


def get_interpolation_weights(old_size, new_size):
    """     Calc the neighboor index and coeff
    Args:
        old_size (int): Size of the original image on one axis.
        new_size (int): Size of the new image on the same axis.
    Returns:
        tuple: (i0, i1, weights) where i0 and i1 are
    """
    coords = np.linspace(0, old_size - 1, new_size)
    i0 = np.floor(coords).astype(int)
    i1 = np.minimum(i0 + 1, old_size - 1)
    weights = coords - i0
    return i0, i1, weights


def interpolate_rows(image, x0, x1, dx):
    """     Interpolation horizontale (cols).
    Args:
        image (numpy array): Matrice of points of the image.
        x0 (array): Indices des pixels de gauche.
        x1 (array): Indices des pixels de droite.
        dx (): Poids d'interpolation pour les pixels de droite.
    Returns:
        array: Matrice de points de l'image après interpolation horizontale.
    """
    dx_row = dx.reshape(1, -1)
    return image[:, x0] * (1 - dx_row) + image[:, x1] * dx_row


def interpolate_columns(temp_image, y0, y1, dy):
    """     Interpolation verticale (rows).
    Args:
        temp_image (numpy array): Matrice de points de l'image après interpolation horizontale.
        y0 (array): Indices des pixels du haut.
        y1 (array): Indices des pixels du bas.
        dy (): Poids d'interpolation pour les pixels du bas.
    Returns:
        Matrice de points de l'image après interpolation verticale.
    """
    dy_col = dy.reshape(-1, 1)
    return temp_image[y0] * (1 - dy_col) + temp_image[y1] * dy_col


def bilinear_interpolation(image, new_shape):
    """     Resize a image with inerpolation bilinear by decomposing the operation into two linear steps.
    Args:
        image (numpy array): Matrice of points of the image.
        new_shape (tuple): New shape of the image (new_height, new_width).
    Returns:
        Matrice of points of the image resized.
    """
    h_old, w_old = image.shape
    h_new, w_new = new_shape
    y0, y1, dy = get_interpolation_weights(h_old, h_new)
    x0, x1, dx = get_interpolation_weights(w_old, w_new)
    row_interpolated = interpolate_rows(image, x0, x1, dx)
    result = interpolate_columns(row_interpolated, y0, y1, dy)
    return result.astype(np.float32)


def resize_image(image, size_x, size_y):
    """     Resize the image to the new size.
    Args:
        image (numpy array): Matrice of points of the image.
        size_x (int): New width of the image.
        size_y (int): New height of the image.
    Returns:
        Matrice of points of the image resized.
    """
    # return np.resize(image, (size_x, size_y))
    return bilinear_interpolation(image, (size_x, size_y))


def normalize_image(image):
    """     Normalize the image by dividing each pixel value by the maximum pixel value.
    Args:
        image (numpy array): Matrice of points of the image.
    Returns:
        Matrice of points of the image normalized.
    """
    if image.max() > 1.0:
        image /= 255.0
    return image


# This function is no longer used, but it can be useful for future use if we want to add more preprocessing steps
# It was replaced by a sklearn pipeline
def preprocess_image(image, size=(224, 224)):
    image = image_to_array(image)
    image = divide_gray_scale(image)
    image = resize_image(image, size[0], size[1])
    image = normalize_image(image)
    return image


def flatten_image_array(matrix_array):
    """     Flatten the image array to a 1D array.
    Args:
        matrix_array (numpy array): Matrice of points of the image.
    Returns:
        Matrice of points of the image flattened.
    """
    n_samples = matrix_array.shape[0]
    flat_matrix_array = matrix_array.reshape(n_samples, -1)
    return flat_matrix_array
