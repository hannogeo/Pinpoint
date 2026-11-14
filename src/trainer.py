import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .config import (DATA_DIR, SAMPLES_PATH, MODEL_PATH, META_PATH, DB_PATH,
                     IMAGES_DIR)
from .model import build_model, save_meta, save_db
from .dataset import GeoDataset, get_transforms
from .geo import make_cells
from .uncertainty import calibrate_radius


FEATURES_PATH = os.path.join(DATA_DIR, "features.npy")
LABELS_PATH = os.path.join(DATA_DIR, "labels.npy")


def run_training(cfg, progress_cb=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(SAMPLES_PATH) as f:
        samples = json.load(f)
    if not samples:
        raise RuntimeError("No samples. Prepare dataset first.")

    lats = [s["lat"] for s in samples]
    lngs = [s["lng"] for s in samples]
    n_clusters = int(cfg.get("n_clusters", 300))
    labels, centers = make_cells(lats, lngs, n_clusters)
    for i, s in enumerate(samples):
        s["cell"] = int(labels[i])
    with open(SAMPLES_PATH, "w") as f:
        json.dump(samples, f)

    num_classes = len(centers)
    backbone_name = cfg.get("backbone", "efficientnet_b0")
    img_size = int(cfg.get("img_size", 224))

    # Build backbone (pretrained) for feature extraction
    import timm
    backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
    backbone = backbone.to(device).eval()
    feat_dim = backbone.num_features

    n = len(samples)
    bs = int(cfg.get("batch_size", 64))
    nw = int(cfg.get("num_workers", 0))
    epochs = int(cfg.get("epochs", 8))
    total_batches = (n + bs - 1) // bs
    global_total = total_batches + epochs * total_batches

    # Incremental feature extraction: reuse existing, append new
    existing_n = 0
    # Invalidate cache if backbone or image size changed
    meta_old = None
    try:
        import json as _j
        if os.path.exists(META_PATH):
            meta_old = _j.load(open(META_PATH))
    except Exception:
        meta_old = None
    cache_valid = True
    if meta_old is not None:
        if meta_old.get("backbone") != backbone_name or meta_old.get("img_size") != img_size:
            cache_valid = False
    if os.path.exists(FEATURES_PATH) and os.path.exists(LABELS_PATH) and cache_valid:
        try:
            # features.npy is a raw memmap (no .npy header), so the row count
            # comes from the file size rather than from an array shape.
            rows = os.path.getsize(FEATURES_PATH) // (4 * int(feat_dim))
            existing_n = min(rows, n)
        except Exception:
            existing_n = 0
    else:
        existing_n = 0

    if existing_n == n:
        # fully cached
        mm = np.memmap(FEATURES_PATH, dtype="float32", mode="r+", shape=(n, feat_dim))
        # Cells were just recomputed deterministically; the cached *features*
        # are label-independent, so use these labels rather than stale ones.
        label_arr = np.asarray(labels, dtype=np.int64)
        if progress_cb:
            progress_cb(total_batches, total_batches, "Features cached - starting training", "extract")
    else:
        # extract from existing_n to n (append)
        transform = get_transforms(train=False, size=img_size)
        # dataset for new tail
        new_samples = samples[existing_n:]
        dataset = GeoDataset(new_samples, transform=transform)
        loader = DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
        # create combined memmap
        tmp_path = FEATURES_PATH + ".tmp"
        new_mm = np.memmap(tmp_path, dtype="float32", mode="w+", shape=(n, feat_dim))
        new_labels = np.empty(n, dtype=np.int64)
        if existing_n > 0 and os.path.exists(FEATURES_PATH):
            old_mm = np.memmap(FEATURES_PATH, dtype="float32", mode="r", shape=(existing_n, feat_dim))
            new_mm[:existing_n] = old_mm[:]
            old_labels = np.load(LABELS_PATH)
            new_labels[:existing_n] = old_labels[:existing_n]
        idx = existing_n
        total_batches = len(loader)
        global_total = total_batches + epochs * total_batches
        step = 0
        for imgs, cells, _, _ in loader:
            imgs = imgs.to(device, non_blocking=True)
            with torch.no_grad():
                f = backbone(imgs)
                if f.dim() == 4:
                    f = f.mean(dim=(2, 3))
            f = f.cpu().numpy().astype("float32")
            new_mm[idx:idx + len(f)] = f
            new_labels[idx:idx + len(cells)] = cells.numpy()
            idx += len(f)
            step += 1
            if progress_cb:
                progress_cb(step, total_batches, f"Extracting features: {step}/{total_batches}", "extract")
        new_mm.flush()
        del new_mm
        if 'old_mm' in locals():
            del old_mm
        import gc
        gc.collect()
        if os.path.exists(FEATURES_PATH):
            try:
                os.remove(FEATURES_PATH)
            except Exception:
                pass
        import time
        for _attempt in range(5):
            try:
                os.replace(tmp_path, FEATURES_PATH)
                break
            except Exception:
                time.sleep(1)
        else:
            raise
        np.save(LABELS_PATH, new_labels)
        mm = np.memmap(FEATURES_PATH, dtype="float32", mode="r+", shape=(n, feat_dim))
        label_arr = np.asarray(labels, dtype=np.int64)
    X = torch.from_numpy(mm).float()
    y = torch.from_numpy(label_arr).long()

    # Class weights (inverse sqrt frequency) keep rarely visited cells visible.
    counts = np.bincount(label_arr, minlength=num_classes).astype(np.float64)
    weights = 1.0 / np.sqrt(counts + 1.0)
    weights = weights / weights.mean()
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    head_type = cfg.get("head_type", "arcface")
    emb_dim = int(cfg.get("emb_dim", 512))
    arc_scale = float(cfg.get("arcface_scale", 30.0))
    arc_margin = float(cfg.get("arcface_margin", 0.2))
    lr = float(cfg.get("lr", 1e-3))

    train_ds = TensorDataset(X, y)
    # Head training uses cached features, so big batches are both fast and
    # stable; the extraction batch size stays small for image loading.
    train_bs = int(cfg.get("train_batch_size", 4096))
    train_loader = DataLoader(train_ds, batch_size=train_bs, shuffle=True, num_workers=0, pin_memory=True)

    step = 0  # head-training steps
    train_total = epochs * max(1, len(train_loader))

    if head_type == "arcface":
        # Metric-learning head: project the frozen backbone features into a
        # compact, geographically meaningful embedding (CosFace margin), so
        # the kNN database is organised by location instead of by looks.
        proj = nn.Linear(feat_dim, emb_dim, bias=False).to(device)
        weight = nn.Parameter(torch.empty(num_classes, emb_dim, device=device))
        nn.init.xavier_uniform_(weight)
        params = list(proj.parameters()) + [weight]
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        for epoch in range(epochs):
            proj.train()
            running_loss = 0.0
            for batch_idx, (xb, yb) in enumerate(train_loader):
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                z = torch.nn.functional.normalize(proj(xb), dim=1)
                cos = z @ torch.nn.functional.normalize(weight, dim=1).t()
                # Additive angular margin: pulls same-cell samples together and
                # pushes everything else apart, which is what retrieval needs.
                cos = cos - arc_margin * torch.nn.functional.one_hot(
                    yb, num_classes).float()
                loss = torch.nn.functional.cross_entropy(
                    arc_scale * cos, yb, weight=class_weights)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                running_loss += loss.item()
                step += 1
                if progress_cb:
                    progress_cb(step, train_total, f"Training epoch {epoch+1}/{epochs} - loss {loss.item():.4f}", "train")
            scheduler.step()
            avg_loss = running_loss / max(1, len(train_loader))
            if progress_cb:
                progress_cb(step, train_total, f"Epoch {epoch+1}/{epochs} done - avg loss {avg_loss:.4f}", "train")
    else:
        head = nn.Linear(feat_dim, num_classes).to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        for epoch in range(epochs):
            head.train()
            running_loss = 0.0
            for batch_idx, (xb, yb) in enumerate(train_loader):
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = head(xb)
                loss = criterion(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                optimizer.step()
                running_loss += loss.item()
                step += 1
                if progress_cb:
                    progress_cb(step, train_total, f"Training epoch {epoch+1}/{epochs} - loss {loss.item():.4f}", "train")
            scheduler.step()
            avg_loss = running_loss / max(1, len(train_loader))
            if progress_cb:
                progress_cb(step, train_total, f"Epoch {epoch+1}/{epochs} done - avg loss {avg_loss:.4f}", "train")

    # Build full model (backbone + trained head) and save
    model, _ = build_model(num_classes, backbone=backbone_name, pretrained=False,
                           emb_dim=emb_dim, head_type=head_type, scale=arc_scale)
    model.backbone.load_state_dict(backbone.state_dict())
    if head_type == "arcface":
        model.proj.load_state_dict(proj.state_dict())
        model.weight.data.copy_(weight.data)
    else:
        model.head.load_state_dict(head.state_dict())
    model = model.to(device).eval()

    torch.save({"state_dict": model.state_dict(), "num_classes": num_classes,
                "backbone": backbone_name, "feat_dim": feat_dim, "img_size": img_size,
                "head_type": head_type, "emb_dim": int(model.emb_dim),
                "arcface_scale": arc_scale}, MODEL_PATH)
    meta = {"centers": centers.tolist(), "num_classes": num_classes,
            "backbone": backbone_name, "img_size": img_size,
            "head_type": head_type, "emb_dim": int(model.emb_dim),
            "arcface_scale": arc_scale}
    save_meta(meta)

    # Embedding DB from cached features (normalized, in the trained space).
    embeddings = np.empty((n, int(model.emb_dim)), dtype=np.float32)
    for start in range(0, n, 8192):
        stop = min(start + 8192, n)
        with torch.no_grad():
            z = model.embed_features(X[start:stop].to(device, non_blocking=True))
        embeddings[start:stop] = z.cpu().numpy()
    lats_arr = np.array(lats, dtype=np.float64)
    lngs_arr = np.array(lngs, dtype=np.float64)
    cells_arr = np.array(labels, dtype=np.int64)
    save_db(embeddings, lats_arr, lngs_arr, cells_arr)

    # Calibrate the uncertainty radius so the reported 80% radius really covers
    # ~80% of held-out panoramas (leave-one-panorama-out evaluation).
    if progress_cb:
        progress_cb(train_total, train_total,
                    "Calibrating confidence...", "train")

    def _probs_for(emb_batch):
        with torch.no_grad():
            z = torch.from_numpy(np.ascontiguousarray(emb_batch)).to(device)
            logits = model.logits_from_emb(z)
            return torch.softmax(logits, dim=1).cpu().numpy()

    try:
        pano_ids = [s.get("panoId") or f"sample-{i}" for i, s in enumerate(samples)]
        calib = calibrate_radius(
            embeddings, lats_arr, lngs_arr, cells_arr, pano_ids, _probs_for,
            sample_n=int(cfg.get("calib_samples", 400)),
        )
    except Exception:
        calib = None
    if calib is not None:
        meta["radius_scale"] = calib["scale"]
        meta["calibration"] = {k: v for k, v in calib.items()
                               if k not in ("confidence_x", "confidence_y")}
        if calib.get("confidence_x"):
            meta["confidence_map"] = {"x": calib["confidence_x"],
                                      "y": calib["confidence_y"]}
        save_meta(meta)

    if progress_cb:
        progress_cb(train_total, train_total, "training complete", "train")
    return {"num_classes": num_classes, "n_samples": len(samples)}

