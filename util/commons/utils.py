import numpy as np
import scipy.spatial as ssp
import scipy.stats as sst


def get_distance_matrix(points, world_size=None, torus=False, add_to_diagonal=0):
    distance_matrix = np.vstack([get_distances(points, p, torus=torus, world_size=world_size) for p in points])
    distance_matrix = distance_matrix + np.diag(add_to_diagonal * np.ones(points.shape[0]))
    return distance_matrix


def get_distances(x0, x1, torus=False, world_size=None):
    delta = np.abs(x0 - x1)
    if torus:
        delta = np.where(delta > world_size / 2, delta - world_size, delta)
    dist = np.sqrt((delta ** 2).sum(axis=-1))
    return dist


def get_euclid_distances(points, matrix=True):
    if matrix:
        dist = ssp.distance.squareform(
            ssp.distance.pdist(points, 'euclidean'))
    else:
        dist = ssp.distance.pdist(points, 'euclidean')
    return dist


def skip_diag_strided(A):
    m = A.shape[0]
    strided = np.lib.stride_tricks.as_strided
    s0,s1 = A.strides
    return strided(A.ravel()[1:], shape=(m-1,m), strides=(s0+s1,s1)).reshape(m,-1)