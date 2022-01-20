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
        raise NotImplementedError('The data type is not implemented! Must be array or pandas dataframe!')

    xLocation = array[:, 0].astype(int)
    yLocation = array[:, 1].astype(int)
    xLocation = xLocation - min(xLocation)
    yLocation = yLocation - min(yLocation)
    col = max(np.unique(xLocation))
    row = max(np.unique(yLocation))
    im = np.zeros((col, row))
    im[:] = fill
    for i in range(len(xLocation)):
        im[np.asscalar(xLocation[i]) - 1, np.asscalar(yLocation[i] - 1)] = array[i, 2]
    return im


class CorSolver:
    """
    A class that takes tie points from two coordinate system (i.e., one from Flex, the other from xray measurement)
    and solve the transformation between these two systems.
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
        l = len(src_tri)
        B = np.vstack([np.transpose(src_tri), np.ones(l)])
        D = 1.0 / np.linalg.det(B)

        def entry(r, d):
            return np.linalg.det(np.delete(np.vstack([r, B]), (d + 1), axis=0))

        M = [[(-1) ** i * D * entry(R, i) for i in range(l)] for R in np.transpose(dst_tri)]
        A, t = np.hsplit(np.array(M), [l - 1])
        t = np.transpose(t)[0]
        self.transformation_matrix = A
        self.translation_vector = t
        return self

    def transform(self, src_coordinate):
        """
        Parameters:
        --------
            src_coordinate: the source coordinates that needs be transformed
        """
        dst_coordinate = src_coordinate.dot(self.transformation_matrix.T) + self.translation_vector
        return np.round(dst_coordinate, 0)


