import numpy as np
from scipy.spatial import cKDTree

def fusePoints(points, intensities=None, distance=5):
    tree = cKDTree(points)
    rows_to_fuse = tree.query_pairs(r=distance)

    for (r1, r2) in rows_to_fuse:
        points[r1] = (points[r1] + points[r2])//2

    duplicates = [r2 for (r1, r2) in rows_to_fuse]
    mask = np.ones(len(points), dtype=bool)
    mask[duplicates] = False
    
    if intensities is not None:
        return points[mask,:], intensities[mask,:]
    else:
        return points[mask,:]