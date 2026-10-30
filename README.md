# Finder

A desktop (Streamlit) app that trains an image->region model from Google Street View panoramas
identified by `panoId`, then guesses the location of an uploaded image and pins it on an OSM map.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```
or on Windows: double-click `run.bat`.

## Workflow

1. Training tab: upload one or more JSON files in the documented format.
2. Click "Prepare dataset (download panos)" to fetch panoramas via `panoId` and render perspective views.
3. Adjust training settings and click "Train model". Uses GPU (CUDA) + mixed precision when available.
4. Usage tab: upload an image, click "Predict location". Shows predicted coordinates, confidence, and an OSM map pin.

## Notes

- Panoramas are fetched without the Google API using the `streetlevel` library.
- Training data and models are stored under `data/` and `models/`.
- For best results, provide thousands of locations. More geocells = finer regions.
