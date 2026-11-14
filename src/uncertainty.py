import math

import numpy as np

R_EARTH_KM = 6371.0

# Retrieval weighting
SIM_TEMP = 0.05          # softmax temperature over cosine similarities
PRIOR_BETA = 0.5         # how strongly the classifier's cell prior matters
MAX_CANDIDATES = 2000    # neighbours considered for the pin / uncertainty
MIN_RADIUS_KM = 1.0      # never claim more precision than this
TOP_CELLS = 5            # retrieval is restricted to the most likely cells
CELL_PRIOR_FLOOR = 1e-3  # minimum cell prior, so exact matches still win
EXACT_MATCH_SIM = 0.92   # above this, skip the cell restriction entirely


def haversine_km(lat1, lng1, lat2, lng2):
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R_EARTH_KM * math.asin(math.sqrt(min(1.0, max(0.0, a))))


def _haversine_vec(lat, lng, lats, lngs):
    p1 = math.radians(lat)
    p2 = np.radians(lats)
    dphi = np.radians(np.asarray(lats) - lat)
    dl = np.radians(np.asarray(lngs) - lng)
    a = np.sin(dphi / 2) ** 2 + math.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R_EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _to_xyz(lat, lng):
    latr = np.radians(lat)
    lngr = np.radians(lng)
    return np.stack(
        [np.cos(latr) * np.cos(lngr), np.cos(latr) * np.sin(lngr), np.sin(latr)],
        axis=-1,
    )


def _from_xyz(vec):
    vec = np.asarray(vec, dtype=np.float64)
    vec = vec / (np.linalg.norm(vec) + 1e-12)
    return (float(np.degrees(np.arcsin(np.clip(vec[2], -1.0, 1.0)))),
            float(np.degrees(np.arctan2(vec[1], vec[0]))))


def _weighted_quantile(values, weights, level):
    """Smallest value whose cumulative weight reaches ``level``."""
    order = np.argsort(values)
    v = np.asarray(values)[order]
    c = np.cumsum(np.asarray(weights)[order])
    k = int(np.searchsorted(c, level, side="left"))
    k = min(max(k, 0), len(v) - 1)
    return float(v[k]), order, c


def spatial_stats(embeddings, lats, lngs, cells, probs, query=None, sims=None,
                  exclude=None, max_candidates=MAX_CANDIDATES,
                  sim_temp=SIM_TEMP, prior_beta=PRIOR_BETA,
                  min_radius_km=MIN_RADIUS_KM):
    """Weighted spatial summary of the training locations that best match a query.

    The pin is the weighted mean of the dominant cluster (not of every far-away
    low-weight match), and the radius is a weighted quantile of the real
    great-circle distances, so a handful of stray matches cannot blow it up.
    """
    if sims is None:
        if query is None:
            raise ValueError("either query or sims is required")
        sims = embeddings @ query
    sims = np.asarray(sims)

    n = len(sims)
    if n == 0:
        return None
    if exclude is not None:
        sims = sims.copy()
        sims[np.asarray(exclude, dtype=bool)] = -np.inf

    m = int(min(max_candidates, n))
    idx = np.argpartition(sims, -m)[-m:] if m < n else np.arange(n)
    s = sims[idx]
    if not np.isfinite(s).any():
        return None

    if probs is not None and cells is not None:
        cell_idx = np.clip(np.asarray(cells)[idx].astype(np.int64), 0, len(probs) - 1)
        prior = np.maximum(np.asarray(probs)[cell_idx], 1e-9) ** prior_beta
    else:
        prior = np.ones(m, dtype=np.float64)

    w = np.exp((s - np.max(s)) / sim_temp) * prior
    total = float(w.sum())
    if not np.isfinite(total) or total <= 0:
        w = np.ones(m, dtype=np.float64) / m
    else:
        w = w / total

    p_lat = np.asarray(lats, dtype=np.float64)[idx]
    p_lng = np.asarray(lngs, dtype=np.float64)[idx]

    # 1. Rough mode, to anchor the dominant cluster.
    j0 = int(np.argmax(w))
    lat0, lng0 = float(p_lat[j0]), float(p_lng[j0])

    # 2. Core = points inside the radius holding 60% of the weight.
    d0 = _haversine_vec(lat0, lng0, p_lat, p_lng)
    r_core, _, _ = _weighted_quantile(d0, w, 0.6)
    core = d0 <= max(r_core, min_radius_km)
    if not core.any():
        core = np.zeros(m, dtype=bool)
        core[j0] = True

    wc = w[core]
    wc = wc / wc.sum()
    xyz = _to_xyz(p_lat[core], p_lng[core])
    centroid = (xyz * wc[:, None]).sum(axis=0)
    if np.linalg.norm(centroid) < 1e-6:
        pin_lat, pin_lng = lat0, lng0
    else:
        pin_lat, pin_lng = _from_xyz(centroid)

    # 3. Real distances from the pin; radii are weighted quantiles.
    d = _haversine_vec(pin_lat, pin_lng, p_lat, p_lng)
    r50 = _weighted_quantile(d, w, 0.50)[0]
    r80 = _weighted_quantile(d, w, 0.80)[0]
    r95 = _weighted_quantile(d, w, 0.95)[0]

    # Sorted distances / cumulative weight, so callers can ask "what fraction
    # of the probability mass is within R km?".
    order = np.argsort(d)
    dist_sorted = d[order]
    cum_w = np.cumsum(w[order])

    # 4. Uncertainty ellipse from the dominant cluster, scaled to r80.
    ellipse = None
    if core.sum() >= 3:
        try:
            latr = math.radians(pin_lat)
            east = ((p_lng[core] - pin_lng + 180.0) % 360.0 - 180.0) * math.cos(latr) * 111.320
            north = (p_lat[core] - pin_lat) * 110.574
            me = float((wc * east).sum())
            mn = float((wc * north).sum())
            de, dn = east - me, north - mn
            var_ee = float((wc * de * de).sum())
            var_nn = float((wc * dn * dn).sum())
            var_en = float((wc * de * dn).sum())
            cov = np.array([[var_ee, var_en], [var_en, var_nn]])
            eigvals, eigvecs = np.linalg.eigh(cov)
            order_e = np.argsort(eigvals)[::-1]
            lam1 = max(float(eigvals[order_e[0]]), 1e-9)
            lam2 = max(float(eigvals[order_e[1]]), 0.0)
            axis = eigvecs[:, order_e[0]]
            minor = r80 * math.sqrt(lam2 / lam1)
            bearing = (90.0 - math.degrees(math.atan2(axis[1], axis[0]))) % 180.0
            ellipse = {
                "major_km": float(r80),
                "minor_km": float(minor),
                "bearing_deg": float(bearing),
            }
        except Exception:
            ellipse = None

    return {
        "lat": float(pin_lat),
        "lng": float(pin_lng),
        "r50_km": max(float(r50), min_radius_km),
        "r80_km": max(float(r80), min_radius_km),
        "r95_km": max(float(r95), min_radius_km),
        "dist_km": dist_sorted,
        "cum_w": cum_w,
        "ellipse": ellipse,
        "n_candidates": int(m),
    }


def mass_within(stats, radius_km):
    """Fraction of the retrieval weight closer than ``radius_km`` to the pin."""
    if stats is None:
        return 0.0
    d = stats["dist_km"]
    c = stats["cum_w"]
    k = int(np.searchsorted(d, radius_km, side="right"))
    return float(c[k - 1]) if k > 0 else 0.0


def mask_cell_probs(probs, top_cells=TOP_CELLS, floor=CELL_PRIOR_FLOOR):
    """Keep only the most likely cells, but never zero the rest completely:
    retrieval inside the top cells is much more reliable, while a floor lets a
    near-exact match outside those cells still win."""
    p = np.asarray(probs, dtype=np.float64)
    if not top_cells or top_cells >= len(p):
        return p
    keep = np.argpartition(p, -top_cells)[-top_cells:]
    out = np.full_like(p, floor)
    out[keep] = np.maximum(p[keep], floor)
    return out


def _isotonic_map(values, outcomes, bins=8):
    """Monotone piecewise-linear estimate of P(outcome | value)."""
    values = np.asarray(values, dtype=np.float64)
    outcomes = np.asarray(outcomes, dtype=np.float64)
    if len(values) < 40:
        return [], []
    order = np.argsort(values)
    v, o = values[order], outcomes[order]
    edges = np.linspace(0, len(v), bins + 1).astype(int)
    xs, ys = [], []
    for k in range(bins):
        lo, hi = edges[k], edges[k + 1]
        if hi <= lo:
            continue
        xs.append(float(v[lo:hi].mean()))
        ys.append(float(o[lo:hi].mean()))
    for k in range(1, len(ys)):
        ys[k] = max(ys[k], ys[k - 1])
    return xs, ys


def calibrate_radius(embeddings, lats, lngs, cells, pano_ids, probs_for,
                     sample_n=400, batch=32, seed=0,
                     top_cells=TOP_CELLS, sim_temp=SIM_TEMP):
    """Measure how much the 80% radius (and the confidence) has to be adjusted.

    A sample of panoramas is located using every *other* panorama (all views of
    the same pano are excluded).  Comparing the real error with what the model
    would have reported gives:
      * a scale factor that makes the 80% radius honest (80% coverage), and
      * a monotone map from retrieval mass to the observed hit rate, which is
        what the confidence number should report.
    """
    n = len(embeddings)
    if n < 100:
        return None

    pano = np.asarray(pano_ids)
    rng = np.random.default_rng(seed)
    count = int(min(sample_n, n))
    idxs = rng.choice(n, size=count, replace=False)

    ratios, errors, masses, hits = [], [], [], []
    for start in range(0, count, batch):
        chunk = idxs[start:start + batch]
        Q = np.ascontiguousarray(embeddings[chunk], dtype=np.float32)
        sims_all = embeddings @ Q.T  # (n, q)
        probs = probs_for(Q)
        for j, i in enumerate(chunk):
            st = spatial_stats(
                embeddings, lats, lngs, cells,
                mask_cell_probs(probs[j], top_cells),
                sims=sims_all[:, j], exclude=(pano == pano[i]),
                sim_temp=sim_temp,
            )
            if st is None:
                continue
            err = haversine_km(float(lats[i]), float(lngs[i]), st["lat"], st["lng"])
            r80 = st["r80_km"]
            if r80 <= 0:
                continue
            ratios.append(err / r80)
            errors.append(err)
            masses.append(mass_within(st, 100.0))
            hits.append(1.0 if err <= 100.0 else 0.0)

    if len(ratios) < 20:
        return None

    ratios = np.asarray(ratios, dtype=np.float64)
    errors = np.asarray(errors, dtype=np.float64)
    scale = float(np.clip(np.quantile(ratios, 0.8), 0.1, 50.0))
    cx, cy = _isotonic_map(masses, hits, bins=8)
    return {
        "scale": scale,
        "n": int(len(ratios)),
        "median_error_km": float(np.median(errors)),
        "p80_error_km": float(np.quantile(errors, 0.8)),
        "within_25km": float(np.mean(errors <= 25.0)),
        "within_100km": float(np.mean(errors <= 100.0)),
        "within_500km": float(np.mean(errors <= 500.0)),
        "median_raw_ratio": float(np.median(ratios)),
        "target_coverage": 0.8,
        "confidence_x": cx,
        "confidence_y": cy,
    }

