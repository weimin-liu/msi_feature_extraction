import numpy as np


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
