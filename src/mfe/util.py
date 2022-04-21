"""
this module contains utility functions for the mfe package
"""
import numpy as np
import pandas as pd


def imshow(array: np.ndarray, fill=0) -> np.ndarray:
    """

    Args:
        array: array with a shape of (,3) to be pivoted to an image.

        fill: values to fill missing values

    Returns:
        im: image matrix

    """
    if isinstance(array, np.ndarray):
        pass
    elif isinstance(array, pd.DataFrame):
        array = array.to_numpy()
    else:
        raise NotImplementedError('The data type is not implemented! Must be array or pandas '
                                  'dataframe!')

    x_location = array[:, 0].astype(int)
    y_location = array[:, 1].astype(int)
    x_location = x_location - min(x_location)
    y_location = y_location - min(y_location)
    col = max(np.unique(x_location))
    row = max(np.unique(y_location))
    image_arr = np.zeros((col, row))
    image_arr[:] = fill
    for i, _ in enumerate(x_location):
        image_arr[np.asscalar(x_location[i]) - 1, np.asscalar(y_location[i] - 1)] = array[i, 2]
    return image_arr


class CorSolver:
    """
    A class that takes tie points from two coordinate system (i.e., one from Flex, the other from
    xray measurement) and solve the transformation between these two systems.
    """

    def __init__(self):
        self.translation_vector = None
        self.transformation_matrix = None

    def _reset(self):
        if hasattr(self, 'transformation_matrix'):
            del self.transformation_matrix
            del self.translation_vector

    def fit(self, src_tri, dst_tri):
        """
        Parameters:
        --------
            src_tri: coordinates of the source triangle in the source coordinate system
            dst_tri:  coordinates of the target triangle in the target coordinate system
        """
        self._reset()
        return self.partial_fit(src_tri, dst_tri)

    def partial_fit(self, src_tri, dst_tri):
        """
        solve the affine transformation matrix between FlexImage coordinates and X_ray coordinates
        https://stackoverflow.com/questions/56166088/how-to-find-affine-transformation-matrix-between-two-sets-of-3d-points
        """
        len_source = len(src_tri)
        basis = np.vstack([np.transpose(src_tri), np.ones(len_source)])
        diagonal = 1.0 / np.linalg.det(basis)

        def entry(r_val, d_val):
            return np.linalg.det(np.delete(np.vstack([r_val, basis]), (d_val + 1), axis=0))

        m_matrix = [[(-1) ** i * diagonal * entry(R, i) for i in range(len_source)] for R in
                    np.transpose(dst_tri)]
        # pylint: disable=unbalanced-tuple-unpacking
        a_matrix, t_val = np.hsplit(np.array(m_matrix), [len_source - 1])
        t_val = np.transpose(t_val)[0]
        self.transformation_matrix = a_matrix
        self.translation_vector = t_val
        return self

    def transform(self, src_coordinate):
        """
        Parameters:
        --------
            src_coordinate: the source coordinates that needs be transformed
        """
        dst_coordinate = src_coordinate.dot(self.transformation_matrix.T) + self.translation_vector
        return np.round(dst_coordinate, 0)
