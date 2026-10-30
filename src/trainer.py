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
            mm0 = np.memmap(FEATURES_PATH, dtype="float32", mode="r")
            if mm0.shape[1] == feat_dim:
                existing_n = min(mm0.shape[0], n)
                if existing_n == n:
                    existing_n = n  # fully cached
        except Exception:
            existing_n = 0
        try:
            del mm0
        except Exception:
            pass
    else:
        existing_n = 0

    if existing_n == n:
        # fully cached
        mm = np.memmap(FEATURES_PATH, dtype="float32", mode="r+", shape=(n, feat_dim))
        label_arr = np.load(LABELS_PATH)[:n]
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
                progress_cb(step, total_batches, f"extracting features {step}/{total_batches}")
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
        label_arr = np.load(LABELS_PATH)
    X = torch.from_numpy(mm).float()
    y = torch.from_numpy(label_arr).long()

    # Class weights
    counts = np.bincount(label_arr, minlength=num_classes).astype(np.float64)
    weights = 1.0 / np.sqrt(counts + 1.0)
    weights = weights / weights.mean()
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    # Train linear head
    head = nn.Linear(feat_dim, num_classes).to(device)
    epochs = int(cfg.get("epochs", 8))
    lr = float(cfg.get("lr", 1e-3))
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_ds = TensorDataset(X, y)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0, pin_memory=True)

    total_steps = epochs * len(train_loader)
    step = 0
    for epoch in range(epochs):
        head.train()
        running_loss = 0.0
        for xb, yb in train_loader:
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
                progress_cb(step, total_steps, f"epoch {epoch+1}/{epochs} loss {loss.item():.4f}")
        scheduler.step()
        avg_loss = running_loss / max(1, len(train_loader))
        if progress_cb:
            progress_cb(step, total_steps, f"epoch {epoch+1}/{epochs} done avg_loss {avg_loss:.4f}")

    # Build full model (backbone + trained head) and save
    model, _ = build_model(num_classes, backbone=backbone_name, pretrained=False)
    model.backbone.load_state_dict(backbone.state_dict())
    model.head.load_state_dict(head.state_dict())
    model = model.to(device)

    torch.save({"state_dict": model.state_dict(), "num_classes": num_classes,
                "backbone": backbone_name, "feat_dim": feat_dim, "img_size": img_size}, MODEL_PATH)
    meta = {"centers": centers.tolist(), "num_classes": num_classes,
            "backbone": backbone_name, "img_size": img_size}
    save_meta(meta)

    # Embedding DB from cached features (normalized)
    embeddings = torch.nn.functional.normalize(X, dim=1).numpy().astype("float32")
    lats_arr = np.array(lats, dtype=np.float64)
    lngs_arr = np.array(lngs, dtype=np.float64)
    cells_arr = np.array(labels, dtype=np.int64)
    save_db(embeddings, lats_arr, lngs_arr, cells_arr)

    if progress_cb:
        progress_cb(total_steps, total_steps, "training complete")
    return {"num_classes": num_classes, "n_samples": len(samples)}
