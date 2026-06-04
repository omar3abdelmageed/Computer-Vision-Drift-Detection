# Computer Vision Model Monitoring Dashboard

This project monitors the performance and reliability of computer vision models after they are registered for use. It is designed for YOLO object detection and image classification models, with a Streamlit interface and Supabase-backed persistence.

The goal is to make model behavior observable over time: validate the model and dataset before monitoring, build a baseline from training data, process production images, compare current behavior against the baseline, surface drift, raise alerts, and collect human feedback for review-based concept drift.

## App Flow

1. Sign in or create an account.
2. Create or select a model registration.
3. Choose the task type: `object_detection` or `classification`.
4. Register a local YOLO dataset directory.
5. Register a local YOLO `.pt` model artifact.
6. Validate the dataset structure, classes, labels, and image splits.
7. Check model compatibility against the selected dataset.
8. Build a baseline profile from the training and validation data.
9. Configure a watched-folder production image source.
10. Start a monitoring session.
11. Process new images, run YOLO inference, and store predictions.
12. Review predictions, submit feedback, inspect drift charts, and track active alerts.

## Features

- Supabase email/password authentication.
- Model management with name, description, tags, status, and task type.
- YOLO object detection and classification support.
- Local dataset registration and validation.
- Local YOLO `.pt` artifact registration using Ultralytics metadata.
- Model/dataset compatibility checks for task type, class count, and class names.
- Baseline generation from `train` and `val` splits.
- Watched-folder monitoring sessions with start, pause, resume, and stop controls.
- Image deduplication by model, session, and content hash.
- YOLO inference on newly discovered production images.
- Prediction review with approval, rejection, corrected labels, and detection-specific feedback.
- Drift dashboard for data drift, prediction drift, concept drift, and active alerts.
- Session summaries with processed image count, prediction count, confidence, review counts, and duration.

## Dataset Validation

The app validates datasets before they can be used for monitoring.

For object detection datasets, it checks:

- Valid YOLO YAML discovery.
- `nc` and `names` consistency.
- Supported detection layouts using either `images/train` and `labels/train`, or split-first folders such as `train/images` and `train/labels`.
- Required `train` and `val` images.
- Required `train` and `val` labels.
- Matching image and label files.
- YOLO label row format.
- Class IDs within range.
- Normalized bounding box coordinates.

For classification datasets, it checks:

- Split-first class folders under `train`, `val`, and optionally `test`.
- The same structure under `images/train`, `images/val`, and optionally `images/test`.
- Required `train` and `val` images.
- Class folder alignment across validation and test splits when labels are present.

Supported image extensions are `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`, and `.webp`.

## Baseline Outputs

Baseline generation scans the `train` and `val` splits. The `test` split is intentionally excluded.

The baseline stores:

- Dataset split metadata.
- Class names and class counts.
- Image metadata such as dimensions, format, color mode, bit depth, aspect ratio, and file size.
- Image feature summaries.
- Training feature vectors.
- Training class distribution.
- Optional baseline predictions from the registered YOLO model.

## Monitoring Outputs

During a running session, the watched folder is polled for new images. For each new image, the app extracts metadata and image features, runs YOLO inference, stores predictions, and recalculates drift for the latest monitoring window.

For object detection, predictions include:

- Predicted class ID and class name.
- Confidence.
- Normalized bounding box center, width, and height.
- Inference time.

For classification, predictions include:

- Top predicted class ID and class name.
- Confidence.
- Top-k class candidates.
- Inference time.

## Drift Tests And Metrics

The app calculates monitoring output using a combination of image property metrics, distribution tests, and feedback-derived concept metrics.

Data drift:

- RBF-kernel Maximum Mean Discrepancy for total data drift.
- Brightness drift from grayscale pixel intensity.
- Contrast drift from grayscale standard deviation.
- Colour distribution drift from RGB colorfulness.
- Noise drift from grayscale image residuals after light blur.

Prediction drift:

- Chi-Square goodness-of-fit against the fixed training-set class distribution.
- Unseen production classes are treated as critical drift.

Concept drift:

- Human review accuracy from submitted feedback.
- Approval, rejection, corrected label, corrected box, false positive, missed object, and wrong class feedback are stored for review summaries.

Alerts:

- Warning and critical statuses are generated from drift results.
- Active unresolved alerts are shown in the drift dashboard.

## Dependencies Used

Core application:

- `streamlit` for the dashboard UI.
- `supabase` for authentication and persistence.
- `python-dotenv` for environment loading.

Computer vision and inference:

- `ultralytics` for YOLO model metadata and inference.
- `Pillow` for image loading, validation, and drawing overlays.

Data processing and metrics:

- `numpy` for numeric operations.
- `pandas` for tabular display and chart data preparation.
- `scipy` for statistical calculations.
- `PyYAML` for YOLO dataset YAML parsing.
- `altair` for drift visualizations.

Testing:

- `pytest` for the test suite.
- `pytest-cov` for coverage reporting.

Run tests with:

```bash
python -m pytest
```
