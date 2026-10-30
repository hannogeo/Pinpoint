# AGENTS.md — Finder

## Project overview
Finder is a Streamlit app that trains a region-guessing model from Google Street View panoramas
identified by `panoId`, then predicts the location of an uploaded image and pins it on an OSM map.

## Structure
- `app.py` — Streamlit UI (two tabs: Locate, Training).
- `src/` — Python modules:
  - `config.py` — paths and directory creation.
  - `downloader.py` / `downloader_sync.py` — async panorama download + perspective view rendering.
  - `dataset.py` — torch Dataset + transforms.
  - `geo.py` — spherical KMeans geocells.
  - `model.py` — timm backbone + linear head.
  - `trainer.py` — cached-feature training pipeline.
  - `predictor.py` — inference, kNN pin, confidence, estimated error.
  - `maps.py` — static OSM tiles + Leaflet HTML.
- `data/` — **USER DATA. Never edit, delete, or overwrite.** Contains uploaded locations, prepared samples, cached features, images.
- `models/` — **USER DATA.** Trained model weights and embedding DB.

## Critical rules for agents
- **Never modify, delete, or clean anything under `data/` or `models/`.** These hold the user's training data and trained models. Deleting them causes irreversible data loss.
- Do not run cleanup commands that touch `data/` or `models/`.
- Do not commit `data/` or `models/` (see `.gitignore`).
- Keep changes minimal and consistent with existing patterns (Streamlit, torch, timm).

## Running
- Install: `pip install -r requirements.txt`
- Run: `streamlit run app.py` (or `finder` on this machine).

## Commit messages
- Always use Conventional Commits format: `type(scope): short description`
  - Types: feat, fix, docs, style, refactor, perf, test, chore
  - Example: `feat(training): add cached-feature pipeline for scalability`
- Keep the subject under 72 characters; add a body if needed.

## Notes
- GPU (CUDA) is used when available; training uses cached features for scalability.
- Panoramas are fetched without the Google API via the `streetlevel` library.
