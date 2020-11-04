import numpy as np
import scipy.spatial as ssp
import scipy.stats as sst


class EzPickle(object):
    """Objects that are pickled and unpickled via their constructor
    arguments.
    Example usage:
        class Dog(Animal, EzPickle):
            def __init__(self, furcolor, tailkind="bushy"):
                Animal.__init__()
                EzPickle.__init__(furcolor, tailkind)
                ...
    When this object is unpickled, a new Dog will be constructed by passing the provided
    furcolor and tailkind into the constructor. However, philosophers are still not sure
    whether it is still the same dog.
    """

    def __init__(self, *args, **kwargs):
        self._ezpickle_args = args
        self._ezpickle_kwargs = kwargs

    def __getstate__(self):
        return {"_ezpickle_args": self._ezpickle_args, "_ezpickle_kwargs": self._ezpickle_kwargs}

    def __setstate__(self, d):
        out = type(self)(*d["_ezpickle_args"], **d["_ezpickle_kwargs"])
        self.__dict__.update(out.__dict__)


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


def get_adjacency_matrix(distance_matrix, max_dist):
    return np.array((distance_matrix < max_dist) & (distance_matrix > 0.), dtype=float)


def dfs(adj_matrix, minsize):
    """Depth-First-Search，DFS
    
    Arguments:
        adj_matrix {array} -- The adjacency matrix.
        minsize {length} -- [description]
    
    Returns:
        sets -- Returns subsets with at least minsize or more nodes.
    """
    visited = set()
    connected_sets = []

    for ind, row in enumerate(adj_matrix):
        if ind not in visited:
            connected_sets.append(set())
            stack = [ind]
            while stack:
                vertex = stack.pop()
                if vertex not in visited:
                    visited.add(vertex)
                    connected_sets[-1].add(vertex)
                    stack.extend(set(np.where(adj_matrix[vertex, :] != 0)[0]) - visited)
    return [cs for cs in connected_sets if len(cs) >= minsize]

def skip_diag_strided(A):
    m = A.shape[0]
    strided = np.lib.stride_tricks.as_strided
    s0,s1 = A.strides
    return strided(A.ravel()[1:], shape=(m-1,m), strides=(s0+s1,s1)).reshape(m,-1)


def get_connected_sets(sets):
    # find unique directly connected
    # test_sets = [set(np.array(list(s)) % self.nr_actors) for s in sets]
    final_sets = []

    for i, s in enumerate(sets):
        if final_sets:
            if s not in [fs[1] for fs in final_sets]:
                is_super_set = [s >= fs[1] for fs in final_sets]
                if any(is_super_set):
                    del final_sets[np.where(is_super_set)[0][0]]
                is_sub_set = [s <= fs[1] for fs in final_sets]
                if not any(is_sub_set):
                    final_sets.append([i, s])
        else:
            final_sets.append([i, s])

    indices = [fs[0] for fs in final_sets]

    setlist = [list(sets[i]) for i in indices]

    return setlist