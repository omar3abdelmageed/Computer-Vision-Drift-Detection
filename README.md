# Computer Vision Model Monitoring Dashboard

Streamlit + Supabase dashboard for registering YOLO computer vision models, validating YOLO detection/classification datasets, building train/val baselines, running test-folder or local production monitoring sessions, computing drift, raising alerts, and collecting human feedback.

## Setup

1. Create a Python environment.
2. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

3. Copy `secrets/.env.example` to `secrets/.env` and fill in Supabase credentials.
4. Apply the schema migrations in `supabase/migrations/` in order.
5. Make large datasets and YOLO `.pt` artifacts available on the machine running the app. Set `LOCAL_WORKSPACE_DIR` if you want local runtime outputs somewhere other than `workspace/`.
6. Run the app:

```bash
python -m streamlit run streamlit_app.py
```

## Workflow

- Use `Model Registration` to create a registration, choose `object_detection` or `classification`, register a local YOLO dataset directory, register a local `.pt` YOLO artifact, validate compatibility, and build the baseline.
- Use `Live Sessions` to select a baseline-completed registration, configure `test_folder`, `manual_upload`, or `watched_folder` sources, start a session, process images, review predictions/drift/alerts, and submit feedback.

## Drift Tests

The drift layer implements:

- KS, Wasserstein, PSI, Jensen-Shannon, and energy distance for numeric features.
- Jensen-Shannon and class proportion deltas for prediction distributions.
- Centroid distance, nearest-neighbor distance, RBF-MMD fallback, and domain-classifier ROC-AUC for embeddings.
- KSWIN and ADWIN wrappers through River when installed.
- Feedback/ground-truth gated concept drift metrics.

Concept drift is intentionally not calculated unless labels or human feedback exist.
