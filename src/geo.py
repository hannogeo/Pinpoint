import json
import math
import numpy as np
from scipy.cluster.vq import kmeans2


def latlng_to_xyz(lat, lng):
    latr = np.deg2rad(np.asarray(lat, dtype=float))
    lngr = np.deg2rad(np.asarray(lng, dtype=float))
    x = np.cos(latr) * np.cos(lngr)
    y = np.cos(latr) * np.sin(lngr)
    z = np.sin(latr)
    return np.column_stack([x, y, z])


def xyz_to_latlng(xyz):
    xyz = np.asarray(xyz, dtype=float)
    n = np.linalg.norm(xyz, axis=-1, keepdims=True) + 1e-12
    xyz = xyz / n
    lat = np.rad2deg(np.arcsin(np.clip(xyz[..., 2], -1, 1)))
    lng = np.rad2deg(np.arctan2(xyz[..., 1], xyz[..., 0]))
    return lat, lng


def make_cells(lats, lngs, n_clusters):
    lats = np.asarray(lats, dtype=float)
    lngs = np.asarray(lngs, dtype=float)
    xyz = latlng_to_xyz(lats, lngs)
    k = int(min(n_clusters, len(lats)))
    k = max(1, k)
    if k == 1:
        labels = np.zeros(len(lats), dtype=int)
        center_xyz = xyz.mean(axis=0)
        clat, clng = xyz_to_latlng(center_xyz[None, :])
        centers = np.column_stack([clat, clng])
        return labels, centers
    # For very large datasets, cluster on a sample then assign all
    if len(xyz) > 50000:
        step = max(1, len(xyz) // 50000)
        sample = xyz[::step]
        centroids, _ = kmeans2(sample, k, iter=50, minit="++", seed=0)
        # assign all points to nearest centroid
        # compute distances in chunks
        labels = np.empty(len(xyz), dtype=int)
        chunk = 100000
        for i in range(0, len(xyz), chunk):
            sl = slice(i, i + chunk)
            d = np.linalg.norm(xyz[sl, None, :] - centroids[None, :, :], axis=2)
            labels[sl] = np.argmin(d, axis=1)
    else:
        centroids, labels = kmeans2(xyz, k, iter=50, minit="++", seed=0)
    centers_lat, centers_lng = xyz_to_latlng(centroids)
    centers = np.column_stack([centers_lat, centers_lng])
    return labels.astype(int), centers


def cell_for_coords(lat, lng, centers):
    # nearest cell center by spherical distance
    xyz = latlng_to_xyz([lat], [lng])[0]
    cxyz = latlng_to_xyz(centers[:, 0], centers[:, 1])
    d = np.linalg.norm(cxyz - xyz, axis=1)
    return int(np.argmin(d))
