# Finder

A Streamlit app that trains a region-guessing model from Google Street View panoramas
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

1. **Training tab**: upload one or more JSON files (both the original format and the variant with
   `country`/`region`/`imageDate`/etc. are supported). Locations are merged and deduplicated by `panoId`.
2. Click **Train model**. The app automatically downloads panoramas by `panoId`, renders perspective
   views, and trains the model. Progress, elapsed time, and ETA are shown.
3. **Locate tab**: upload an image, click **Predict location**. Shows coordinates (copyable),
   confidence level, estimated error radius, and an interactive OSM map with the pin.

## Notes

- Panoramas are fetched without the Google API using the `streetlevel` library.
- Training data, cached features, and models are stored under `data/` and `models/` (gitignored).
- For best results, provide thousands of locations spread across the target region.
- GPU (CUDA) is used automatically when available.
