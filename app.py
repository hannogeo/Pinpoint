import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import time
import webbrowser
import tempfile

import streamlit as st
import numpy as np
from PIL import Image

from src.config import (DATA_DIR, IMAGES_DIR, MODELS_DIR, LOCATIONS_PATH, SAMPLES_PATH,
                        MODEL_PATH, META_PATH)
from src import geo
from src.downloader import prepare_dataset as prepare_dataset_async
from src.downloader_sync import prepare_dataset_sync
from src.trainer import run_training
from src.predictor import load_model, predict
from src.maps import static_map, leaflet_html, training_map_html

st.set_page_config(page_title="Finder", layout="wide")
st.title("Finder")

# ---- helpers ----
def load_locations():
    if os.path.exists(LOCATIONS_PATH):
        with open(LOCATIONS_PATH) as f:
            return json.load(f)
    return []

def save_locations(locs):
    with open(LOCATIONS_PATH, "w") as f:
        json.dump(locs, f)

def model_exists():
    return os.path.exists(MODEL_PATH)

@st.cache_resource
def cached_model():
    return load_model()

# ---- tabs ----
tab_locate, tab_train = st.tabs(["Locate", "Training"])

# ================= TRAINING TAB =================
with tab_train:
    st.header("Training")
    st.write("Upload JSON files with locations. They will be merged into the training set.")

    uploaded = st.file_uploader("Upload location JSON", type=["json"], accept_multiple_files=True)
    if uploaded:
        processed = st.session_state.get("processed_uploads", set())
        new_files = [uf for uf in uploaded if (uf.name, uf.size) not in processed]
        if new_files:
            locs = load_locations()
            existing_ids = {l.get("panoId") for l in locs}
            added = 0
            for uf in new_files:
                try:
                    data = json.load(uf)
                except Exception as e:
                    st.error(f"Invalid JSON {uf.name}: {e}")
                    continue
                if isinstance(data, list):
                    arr = data
                else:
                    arr = data.get("customCoordinates", [])
                for c in arr:
                    if c.get("panoId") not in existing_ids:
                        locs.append(c)
                        existing_ids.add(c.get("panoId"))
                        added += 1
            save_locations(locs)
            processed = processed | {(uf.name, uf.size) for uf in new_files}
            st.session_state["processed_uploads"] = processed
            st.success(f"Added {added} new locations. Total: {len(locs)}")

    locs = load_locations()
    st.metric("Total locations", len(locs))

    if os.path.exists(SAMPLES_PATH):
        with open(SAMPLES_PATH) as f:
            samps = json.load(f)
        st.metric("Training samples", len(samps))

    if st.button("Train model"):
        if not locs:
            st.warning("No locations loaded.")
        else:
            # Auto-prepare dataset if needed
            need_prep = True
            if os.path.exists(SAMPLES_PATH):
                try:
                    samps = json.load(open(SAMPLES_PATH))
                    sample_ids = {s.get("panoId") for s in samps}
                    loc_ids = {l.get("panoId") for l in locs}
                    if sample_ids == loc_ids and len(samps) > 0:
                        need_prep = False
                except Exception:
                    need_prep = True
            if need_prep:
                progress = st.progress(0)
                status = st.empty()
                prep_start = time.time()
                nc_count = 0
                nc_time = 0.0
                last_call = prep_start
                def pcb(done, total, msg, cached=False):
                    global nc_count, nc_time, last_call
                    now = time.time()
                    if not cached and done > 0:
                        nc_count += 1
                        nc_time += (now - last_call)
                    last_call = now
                    if total > 0:
                        progress.progress(min(1.0, done / total))
                    elapsed = now - prep_start
                    if nc_count > 0 and done > 0:
                        avg_nc = nc_time / nc_count
                        nc_ratio = nc_count / done
                        remaining_nc = (total - done) * nc_ratio
                        eta = avg_nc * remaining_nc
                    else:
                        eta = 0.0
                    status.markdown(
                    f"**{done}/{total} - {msg}**<br>"
                    f"Elapsed: {elapsed:.0f}s &nbsp;|&nbsp; ETA: {eta:.0f}s",
                    unsafe_allow_html=True)
                samples = prepare_dataset_sync(locs, views=4, zoom=2, view_size=224, keep_panos=False, jpeg_quality=85, progress_cb=pcb)
                with open(SAMPLES_PATH, "w") as f:
                    json.dump(samples, f)
                progress.progress(1.0)
                status.text(f"Done. {len(samples)} samples created.")
            cfg = {
                "epochs": 8,
                "batch_size": 32,
                "lr": 1e-3,
                "backbone_lr": 1e-4,
                "n_clusters": min(300, max(50, len(json.load(open(SAMPLES_PATH))) // 40)),
                "img_size": 224,
                "backbone": "efficientnet_b2",
                "freeze_backbone": False,
                "num_workers": 0,
            }
            progress = st.progress(0)
            status = st.empty()
            extract_start = [None]
            train_start = [None]
            def cb(step, total, msg, phase="train"):
                now = time.time()
                if phase == "extract":
                    if extract_start[0] is None:
                        extract_start[0] = now
                    elapsed = now - extract_start[0]
                else:
                    if train_start[0] is None:
                        train_start[0] = now
                    elapsed = now - train_start[0]
                if total > 0:
                    progress.progress(min(1.0, step / total))
                eta = (elapsed / max(1, step)) * (total - step) if step > 0 else 0.0
                status.markdown(
                    f"**{msg}**<br>"
                    f"Elapsed: {elapsed:.0f}s &nbsp;|&nbsp; ETA: {eta:.0f}s &nbsp;|&nbsp; {step}/{total}",
                    unsafe_allow_html=True)
            try:
                with st.spinner("Training..."):
                    res = run_training(cfg, progress_cb=cb)
                progress.progress(1.0)
                status.text("Training complete.")
                st.success(f"Trained. Classes: {res['num_classes']}, samples: {res['n_samples']}")
                cached_model.clear()
            except Exception as e:
                st.error(f"Training failed: {e}")
                import traceback
                st.code(traceback.format_exc())

    st.subheader("Training locations map")
    if st.button("Show training locations map"):
        import streamlit.components.v1 as components
        components.html(training_map_html(locs), height=620)

    try:
        import torch
        st.caption(f"PyTorch {torch.__version__} | CUDA available: {torch.cuda.is_available()} | device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    except Exception:
        pass

# ================= LOCATE TAB =================
with tab_locate:
    st.header("Locate")
    if not model_exists():
        st.warning("No trained model found. Train one first.")
    else:
        uploaded_img = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "webp"])
        if uploaded_img is not None:
            img = Image.open(uploaded_img).convert("RGB")
            import base64, io as _io
            _buf = _io.BytesIO()
            img.save(_buf, format="PNG")
            _b64 = base64.b64encode(_buf.getvalue()).decode()
            col_img, col_info = st.columns([1, 1])
            with col_img:
                import streamlit.components.v1 as components
                components.html(f'''<div style="text-align:center;">
                <img src="data:image/png;base64,{_b64}" style="max-width:480px;width:100%;cursor:zoom-in;border-radius:6px;" onclick="if(this.requestFullscreen) this.requestFullscreen();" title="Click to fullscreen" />
                </div>''', height=400)
            with col_info:
                if st.button("Predict location", use_container_width=True):
                    with st.spinner("Predicting..."):
                        res = predict(img)
                    if res is None:
                        st.error("Model load failed.")
                    else:
                        st.session_state["last_res"] = res
                res = st.session_state.get("last_res")
                if res is not None:
                    st.code(f"{res['lat']:.6f}, {res['lng']:.6f}", language=None)
                    st.write(f"**Confidence:** {res['confidence']*100:.1f}% ({res['confidence_band']})")
                    st.write(f"**Estimated error:** ~{res['estimated_error_km']:.0f} km")
            res = st.session_state.get("last_res")
            if res is not None:
                try:
                    import streamlit.components.v1 as components
                    components.html(leaflet_html(res["lat"], res["lng"]), height=520)
                except Exception as e:
                    st.warning(f"Map render failed: {e}")
