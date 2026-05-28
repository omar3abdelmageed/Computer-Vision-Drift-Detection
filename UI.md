# UI Specification for Computer Vision Model Monitoring Dashboard

## Purpose

This document defines the Streamlit frontend for the Computer Vision Model Monitoring Dashboard.

Use this file as the implementation guide for the app UI, page structure, user workflows, component behavior, status states, and Supabase data dependencies.

The UI must follow the workflow from `cv_monitoring_system_design.md`, but table naming must follow `database.md` whenever the two documents conflict.

Important naming rule:

```text
models = registered monitoring model entity
model_artifacts = uploaded YOLO model file metadata
```

Do not use "Project" or "Create Project" as the user-facing workflow name. The app should present the primary workflow as model registration and live monitoring.

---

## 1. Streamlit App Structure

The Streamlit app must use two primary tabs:

```text
Streamlit App
├── Tab 1: Model Registration
└── Tab 2: Live Sessions
```

Recommended entrypoint:

```text
app/streamlit_app.py
```

Recommended UI module structure:

```text
app/
  streamlit_app.py
  tabs/
    model_registration.py
    live_sessions.py
  components/
    status_cards.py
    charts.py
    image_viewer.py
    feedback_widgets.py
```

The main app should:

1. Load Supabase configuration.
2. Initialize repository or service clients.
3. Render the app title.
4. Render the two main tabs.
5. Keep workflow state in Supabase as the source of truth, with Streamlit session state only for transient UI selections.

---

## 2. Global UI Principles

The UI should be practical, dense, and workflow-oriented. This is an operational monitoring tool, not a marketing page.

Use:

- Status cards for validation, compatibility, baseline, session, drift, and alert state.
- Tables for registered models, datasets, paths, sessions, predictions, drift results, and alerts.
- Charts for baseline distributions, live drift windows, class distributions, confidence distributions, and performance metrics.
- Expanders for detailed validation errors, raw metadata, and JSON payloads.
- Forms for registration metadata, source configuration, and human feedback.
- Buttons for explicit workflow actions such as validate, check compatibility, build baseline, start, pause, resume, stop, and refresh.

Avoid:

- A landing page.
- User-facing "project" terminology.
- Mixing data drift, prediction drift, and concept drift into one undifferentiated score.
- Showing concept drift when no labels or human feedback exist.
- Treating the test split as part of the baseline by default.

---

## 3. Shared Data and Status Model

### 3.1 Core Supabase Tables

The UI should read from and write to these tables:

```text
models
datasets
dataset_paths
model_artifacts
baseline_profiles
production_sources
monitoring_sessions
images
predictions
drift_results
alerts
feedback
```

Additional supporting tables:

```text
ground_truth_labels
```

### 3.2 Registration Status States

The `models.registration_status` field drives the Model Registration workflow.

Supported values:

```text
draft
dataset_uploaded
dataset_validated
artifact_uploaded
compatible
baseline_running
baseline_completed
failed
```

UI behavior:

| Status | Meaning | Primary Next Action |
| --- | --- | --- |
| `draft` | Metadata exists but uploads may be missing | Upload dataset and model |
| `dataset_uploaded` | Dataset ZIP has been uploaded | Validate dataset |
| `dataset_validated` | Dataset validation has completed | Upload or validate model artifact |
| `artifact_uploaded` | YOLO model file metadata exists | Check compatibility |
| `compatible` | Dataset and model are compatible | Build baseline |
| `baseline_running` | Baseline job is running | Show progress and disable conflicting actions |
| `baseline_completed` | Model is ready for live sessions | Show baseline report and eligibility |
| `failed` | A registration step failed | Show error details and retry options |

### 3.3 Baseline Status States

The `models.baseline_status` field drives baseline UI state.

Supported values:

```text
not_started
running
completed
failed
```

### 3.4 Validation Statuses

Use these statuses consistently across dataset validation, task validation, model compatibility, drift results, and alert displays:

```text
valid
warning
invalid
ok
critical
info
```

Recommended visual treatment:

| Status | Display |
| --- | --- |
| `valid`, `ok`, `completed` | Success styling |
| `warning` | Warning styling |
| `invalid`, `failed`, `critical` | Error styling |
| `running` | Spinner or progress styling |
| `draft`, `not_started`, `created` | Neutral styling |

---

## 4. Tab 1: Model Registration

The Model Registration tab is responsible for creating or selecting a registered monitoring model, uploading the reference dataset and YOLO model artifact, validating both, building the baseline, and displaying baseline results.

Render these sections in order:

```text
Model Registration
├── 1. Registration Metadata
├── 2. Task Type Selection
├── 3. Dataset Upload
├── 4. Model Upload
├── 5. Dataset Validation
├── 6. Model Compatibility Check
├── 7. Baseline Builder
└── 8. Baseline Report
```

### 4.1 Registration Metadata

Purpose:

- Create a new `models` row.
- Select an existing registered model for review or continuation.

Widgets:

- Selectbox: existing registered model, ordered by `created_at desc`.
- Text input: registration name.
- Text area: optional description.
- Tag input or comma-separated text input: optional tags.
- Button: `Create registration`.
- Button: `Update metadata`.

Database mapping:

```text
models.name
models.description
models.tags
models.registration_status
models.baseline_status
```

Expected behavior:

- Creating a registration inserts a `models` row with `registration_status = 'draft'`.
- Updating metadata modifies only name, description, and tags.
- The UI should show the selected model ID in a small technical detail area or expander, not as the primary label.

### 4.2 Task Type Selection

Purpose:

- Require the user to explicitly choose the model task type.
- Store the selected task separately from any detected task type.

Widgets:

- Radio or segmented control:
  - `object_detection`
  - `classification`
- Status display for detected task type.
- Status display for task validation.

Database mapping:

```text
models.selected_task_type
models.detected_task_type
models.task_validation_status
```

Expected behavior:

- The selected task type controls dataset validation.
- Auto-detection may warn about mismatches but must not silently override the selected task.
- If selected and detected task types disagree, show a warning and block baseline creation until resolved.

### 4.3 Dataset Upload

Purpose:

- Upload a YOLO dataset ZIP.
- Store the raw ZIP in Supabase Storage.
- Extract it into a local workspace.
- Create or update `datasets` and `dataset_paths` records.

Widgets:

- File uploader accepting `.zip`.
- Button: `Upload dataset`.
- Dataset upload status card.
- Expander: extracted dataset root and discovered files summary.

Storage path:

```text
datasets/{model_id}/raw_upload.zip
```

Database mapping:

```text
datasets.model_id
datasets.storage_path
datasets.dataset_root_path
datasets.dataset_type
datasets.dataset_layout
datasets.has_test_split
datasets.test_has_labels
dataset_paths.dataset_id
dataset_paths.split
dataset_paths.images_path
dataset_paths.labels_path
```

Expected behavior:

- If the ZIP contains exactly one top-level folder and no top-level files, treat that folder as the dataset root.
- Otherwise, use the extraction directory as the dataset root.
- After upload, set `models.registration_status = 'dataset_uploaded'` unless a later valid status already applies.

### 4.4 Model Upload

Purpose:

- Upload the YOLO model artifact, usually `best.pt`.
- Store artifact metadata separately from the registered model entity.

Widgets:

- File uploader accepting `.pt` for MVP.
- Button: `Upload model`.
- Artifact metadata card.
- Expander: raw model metadata.

Storage path:

```text
models/{model_id}/best.pt
```

Database mapping:

```text
model_artifacts.model_id
model_artifacts.storage_path
model_artifacts.local_path
model_artifacts.artifact_name
model_artifacts.artifact_type
model_artifacts.model_task
model_artifacts.class_names
model_artifacts.num_classes
model_artifacts.input_size
model_artifacts.raw_metadata
```

Expected behavior:

- Insert a `model_artifacts` row after upload.
- Load the model with Ultralytics to extract metadata.
- Set `models.registration_status = 'artifact_uploaded'` when appropriate.

### 4.5 Dataset Validation

Purpose:

- Validate the uploaded dataset according to `models.selected_task_type`.
- Display health, detected layout, class names, split paths, and validation issues.

Widgets:

- Button: `Validate dataset`.
- Summary status card.
- Metrics:
  - train images
  - val images
  - test images
  - supported image count
  - unsupported file count
  - number of classes
- Table: split paths from `dataset_paths`.
- Table or expander: validation errors and warnings.
- Expander: YAML candidates for detection datasets.

Database mapping:

```text
datasets.validation_status
datasets.validation_errors
datasets.dataset_layout
datasets.selected_yaml_path
datasets.yaml_filename
datasets.yaml_candidates
datasets.yaml_content
datasets.class_source
datasets.class_names
datasets.num_classes
datasets.num_train_images
datasets.num_val_images
datasets.num_test_images
```

Expected behavior:

- For object detection, validate YAML, image paths, label paths, YOLO label rows, and image-label basename matching.
- For classification, validate class folders and image files; do not require YAML or `.txt` labels.
- If validation is valid or warning-level acceptable, set `models.registration_status = 'dataset_validated'`.
- If validation is invalid, block compatibility checks and baseline building.

### 4.6 Model Compatibility Check

Purpose:

- Confirm that the uploaded YOLO artifact is compatible with the selected task and validated dataset.

Widgets:

- Button: `Check compatibility`.
- Compatibility status card.
- Class comparison table:
  - dataset class
  - model class
  - match status
- Expander: compatibility details.

Database mapping:

```text
model_artifacts.is_compatible
model_artifacts.compatibility_status
model_artifacts.compatibility_details
models.registration_status
```

Expected behavior:

- Compare selected task type, detected model task, number of classes, and class names.
- Compatibility status may be `valid`, `warning`, or `invalid`.
- `invalid` blocks baseline building.
- `valid` or acceptable `warning` allows baseline building.
- Set `models.registration_status = 'compatible'` when compatibility passes.

### 4.7 Baseline Builder

Purpose:

- Build a reference baseline from train + val only.
- Exclude test split from baseline by default.

Widgets:

- Button: `Build baseline`.
- Progress indicator while baseline is running.
- Baseline status card.
- Expander: baseline job details and logs if available.

Database mapping:

```text
models.baseline_status
models.registration_status
baseline_profiles.model_id
baseline_profiles.artifact_id
baseline_profiles.dataset_id
baseline_profiles.profile_type
baseline_profiles.metrics
```

Expected behavior:

- Enable the button only when:
  - dataset validation is not invalid
  - model compatibility is not invalid
  - selected and detected task types are not in conflict
- Set `models.baseline_status = 'running'` and `models.registration_status = 'baseline_running'` while running.
- On success, write `baseline_profiles` records and set:
  - `models.baseline_status = 'completed'`
  - `models.registration_status = 'baseline_completed'`
- On failure, set:
  - `models.baseline_status = 'failed'`
  - `models.registration_status = 'failed'`

### 4.8 Baseline Report

Purpose:

- Show the baseline profiles, dataset health, model performance, and distributions used for future drift comparisons.

Widgets:

- Summary metrics row.
- Charts for image quality distributions.
- Charts for class distribution.
- Charts for prediction confidence distribution.
- Object detection charts:
  - per-class AP
  - objects per class
  - boxes per image
  - box size distribution
  - box aspect ratio distribution
- Classification charts:
  - confusion matrix
  - images per class
  - class imbalance
  - top-k metrics where available
- Expander: raw baseline profile JSON.

Database mapping:

```text
baseline_profiles.profile_type
baseline_profiles.metrics
```

Expected behavior:

- If no completed baseline exists, show a neutral empty state.
- If baseline is running, show progress and last known status.
- If baseline failed, show failure details and retry affordance.
- If baseline is completed, show a readiness message for Live Sessions.

---

## 5. Tab 2: Live Sessions

The Live Sessions tab is responsible for selecting a completed registration, running initial test evaluation, configuring production sources, controlling monitoring sessions, displaying predictions, showing drift, showing alerts, and collecting human feedback.

Render these sections in order:

```text
Live Sessions
├── 1. Select Registered Model
├── 2. Initial Test Evaluation
├── 3. Production Source Configuration
├── 4. Session Controls
├── 5. Live Prediction Feed
├── 6. Drift Dashboard
├── 7. Alerts
└── 8. Human Feedback
```

### 5.1 Select Registered Model

Purpose:

- Select a model that is eligible for live monitoring.

Widgets:

- Selectbox: registered model.
- Eligibility status card.
- Dataset summary card.
- Baseline summary card.

Eligibility rules:

```text
datasets.validation_status = valid
model_artifacts.compatibility_status != invalid
models.baseline_status = completed
```

Database mapping:

```text
models
datasets
model_artifacts
baseline_profiles
```

Expected behavior:

- Models that are not eligible may be visible but clearly disabled or marked as incomplete.
- The UI should explain which prerequisite is missing.
- Only eligible models can start monitoring sessions.

### 5.2 Initial Test Evaluation

Purpose:

- Use the dataset test split as the first production-like evaluation when available.

Widgets:

- Status card: test split availability.
- Button: `Run initial test evaluation`.
- Metrics area for test performance if labels exist.

Database mapping:

```text
datasets.has_test_split
datasets.test_has_labels
production_sources.source_type
production_sources.mode
monitoring_sessions.status
images
predictions
ground_truth_labels
drift_results
```

Expected behavior:

- If a test split exists, offer initial evaluation.
- Create a `production_sources` row with:
  - `source_type = 'test_folder'`
  - `mode = 'initial_evaluation'`
- Create a `monitoring_sessions` row with:
  - `source_type = 'test_folder'`
  - `status = 'running_test_evaluation'`
- If labels exist, calculate performance metrics, data drift, and prediction drift.
- If labels do not exist, calculate data drift and prediction drift only.
- Do not calculate concept drift without labels or feedback.
- If no test split exists, show:

```text
No test split found. Configure a live production source to begin monitoring.
```

### 5.3 Production Source Configuration

Purpose:

- Configure incoming image sources for live or simulated production monitoring.

MVP supported source types:

```text
test_folder
manual_upload
watched_folder
```

Advanced source types may be shown as disabled or hidden until implemented:

```text
supabase_bucket
rtsp
usb_camera
```

Widgets:

- Selectbox: source type.
- Text input: source URI or local path.
- Text input: optional label URI.
- Number input: polling interval seconds.
- Selectbox: mode.
- Text area or JSON editor: optional config.
- Button: `Save source`.
- Table: existing active sources.

Database mapping:

```text
production_sources.model_id
production_sources.source_type
production_sources.source_uri
production_sources.label_uri
production_sources.mode
production_sources.polling_interval_seconds
production_sources.is_active
production_sources.config
```

Expected behavior:

- Default polling interval is 30 seconds.
- The source can be deactivated instead of deleted.
- Source configuration is required before starting a live monitoring session unless running initial test evaluation.

### 5.4 Session Controls

Purpose:

- Start, pause, resume, stop, and refresh monitoring sessions.

Widgets:

- Selectbox or table: existing sessions.
- Buttons:
  - `Start session`
  - `Pause session`
  - `Resume session`
  - `Stop session`
  - `Refresh results`
- Session status card.
- Session timeline or metadata summary.

Database mapping:

```text
monitoring_sessions.model_id
monitoring_sessions.artifact_id
monitoring_sessions.source_id
monitoring_sessions.source_type
monitoring_sessions.status
monitoring_sessions.started_at
monitoring_sessions.ended_at
```

Supported statuses:

```text
created
waiting_for_live_data
running_test_evaluation
running_live_monitoring
paused
completed
failed
```

Expected behavior:

- Starting a session creates or updates a `monitoring_sessions` row.
- Pausing sets status to `paused`.
- Resuming sets status to `running_live_monitoring`.
- Stopping sets status to `completed` and writes `ended_at`.
- Refresh reloads recent images, predictions, drift results, and alerts from Supabase.

### 5.5 Live Prediction Feed

Purpose:

- Show recent production images and model predictions for the selected session.

Widgets:

- Recent images gallery or table.
- Image preview panel.
- Prediction details panel.
- Detection overlay for object detection.
- Top-k table for classification.
- Inference time metric.

Database mapping:

```text
images.session_id
images.source
images.split
images.storage_path
images.filename
images.image_format
images.width
images.height
images.brightness_mean
images.contrast
images.sharpness
predictions.image_id
predictions.artifact_id
predictions.task_type
predictions.predicted_class_name
predictions.confidence
predictions.top_k
predictions.x_center
predictions.y_center
predictions.width
predictions.height
predictions.inference_time_ms
```

Expected behavior:

- Show latest 50 images by default.
- For object detection, draw bounding boxes with class and confidence.
- For classification, show predicted class, confidence, and top-k predictions.
- Make raw prediction JSON available in an expander.

### 5.6 Drift Dashboard

Purpose:

- Display drift by category and window.

The UI must separate:

```text
Data Drift
Prediction Drift
Concept Drift
```

Widgets:

- Three sub-tabs or columns:
  - Data Drift
  - Prediction Drift
  - Concept Drift
- Status cards by drift category.
- Time series charts for metric values and thresholds.
- Table of recent drift windows.
- Expander for drift details JSON.

Database mapping:

```text
drift_results.model_id
drift_results.session_id
drift_results.window_start
drift_results.window_end
drift_results.num_images
drift_results.drift_type
drift_results.metric_name
drift_results.metric_value
drift_results.threshold
drift_results.status
drift_results.details
```

Expected behavior:

- Drift is shown over windows, not individual images.
- Data drift uses image features and embeddings.
- Prediction drift uses model outputs.
- Concept drift is shown only when labels or feedback exist.
- If concept drift cannot be calculated, show a neutral message:

```text
Concept drift requires labels or human feedback.
```

### 5.7 Alerts

Purpose:

- Display unresolved and historical drift or monitoring alerts.

Widgets:

- Alert count cards by severity.
- Table: unresolved alerts.
- Toggle: show resolved alerts.
- Button: `Mark resolved`.
- Expander: alert details.

Database mapping:

```text
alerts.model_id
alerts.session_id
alerts.severity
alerts.title
alerts.message
alerts.drift_result_id
alerts.is_resolved
alerts.created_at
alerts.resolved_at
```

Expected behavior:

- Unresolved critical alerts should be visually prominent.
- Resolved alerts should be hidden by default.
- Marking an alert resolved updates `is_resolved` and `resolved_at`.

### 5.8 Human Feedback

Purpose:

- Collect reviewer feedback on predictions.
- Support concept drift calculations only when feedback or labels exist.

Widgets for classification MVP:

- Button: `Approve prediction`.
- Button: `Reject prediction`.
- Selectbox: corrected class.
- Text area: comment.
- Button: `Submit feedback`.

Widgets for object detection MVP:

- Button: `Approve image-level prediction`.
- Button: `Reject image-level prediction`.
- Checkboxes:
  - false positive
  - missed object
  - wrong class
- Text area: comment.
- Button: `Submit feedback`.

Database mapping:

```text
feedback.image_id
feedback.prediction_id
feedback.feedback_type
feedback.corrected_payload
feedback.comment
```

Expected behavior:

- Full bounding-box editing is not required for MVP.
- Feedback creates rows in `feedback`.
- Corrected or delayed labels may also create rows in `ground_truth_labels` when implemented.
- New feedback should be available to concept drift calculations.

---

## 6. UI-to-Database Mapping Summary

| UI Area | Tables |
| --- | --- |
| Registration metadata | `models` |
| Task type selection | `models` |
| Dataset upload | `datasets`, `dataset_paths` |
| Dataset validation | `datasets`, `dataset_paths` |
| Model upload | `model_artifacts` |
| Compatibility check | `model_artifacts`, `models` |
| Baseline builder | `models`, `baseline_profiles` |
| Baseline report | `baseline_profiles`, `datasets` |
| Select registered model | `models`, `datasets`, `model_artifacts`, `baseline_profiles` |
| Initial test evaluation | `production_sources`, `monitoring_sessions`, `images`, `predictions`, `ground_truth_labels`, `drift_results` |
| Production source configuration | `production_sources` |
| Session controls | `monitoring_sessions` |
| Live prediction feed | `images`, `predictions` |
| Drift dashboard | `drift_results` |
| Alerts | `alerts`, `drift_results` |
| Human feedback | `feedback`, `predictions`, `images`, `ground_truth_labels` |

---

## 7. MVP Scope

The MVP UI must support:

- Model Registration tab.
- Live Sessions tab.
- Explicit task type selection.
- Dataset ZIP upload.
- YOLO `.pt` model upload.
- Dataset validation for object detection and classification.
- Model compatibility check.
- Baseline creation from train + val.
- Baseline report.
- Initial test-folder evaluation when test split exists.
- Production source configuration for:
  - `test_folder`
  - `manual_upload`
  - `watched_folder`
- Session start, pause, resume, stop, and refresh.
- Live prediction feed.
- Separate data drift, prediction drift, and concept drift displays.
- Alerts table.
- Basic human feedback.

---

## 8. Non-Goals for MVP UI

Do not prioritize:

- Full bounding-box annotation editor.
- Automatic retraining workflow.
- Multi-model comparison dashboard.
- Distributed worker queue management UI.
- Complex RBAC or organization management.
- Edge deployment controls.
- Advanced active learning workflow.
- Production camera tuning UI for RTSP or USB camera sources.

Advanced sources may exist in the database enum set, but the MVP UI should focus on local and simulated production workflows.

---

## 9. Acceptance Criteria

### 9.1 Navigation

- The app opens directly into the Streamlit dashboard.
- The app has exactly two primary tabs:
  - `Model Registration`
  - `Live Sessions`
- No user-facing workflow is named `Create Project`.

### 9.2 Model Registration

- User can create or select a registered model.
- User can enter registration name, description, and tags.
- User can select object detection or classification.
- User can upload a dataset ZIP.
- User can upload a YOLO `.pt` file.
- User can validate the dataset according to selected task type.
- User can check model compatibility.
- User can build a baseline from train + val.
- Test split is excluded from baseline by default.
- User can view a baseline report after baseline completion.

### 9.3 Live Sessions

- User can select an eligible registered model.
- Ineligible models show clear missing prerequisites.
- If a test split exists, user can run initial test evaluation.
- If no test split exists, user is prompted to configure a production source.
- User can configure `test_folder`, `manual_upload`, and `watched_folder`.
- User can start, pause, resume, stop, and refresh sessions.
- User can view recent images and predictions.
- User can view drift results by drift category.
- User can view and resolve alerts.
- User can submit basic human feedback.

### 9.4 Drift and Feedback

- Data drift, prediction drift, and concept drift are displayed separately.
- Drift is shown over windows.
- Concept drift is not shown as calculated unless labels or feedback exist.
- Feedback records are stored for later concept drift calculations.

### 9.5 Naming Consistency

- The UI specification uses `models` for the registered monitoring entity.
- The UI specification uses `model_artifacts` for uploaded YOLO model file metadata.
- The UI does not depend on a `projects` table.

---

## 10. Implementation Notes

- Supabase is the source of truth for persistent workflow state.
- Streamlit session state should only hold transient UI state such as selected model ID, selected session ID, active sub-tab, and unsaved form inputs.
- Long-running actions such as baseline building and monitoring should update Supabase status fields so the UI can recover after refresh.
- Raw JSON metadata should be available in expanders for debugging but should not dominate the main workflow.
- Every destructive or state-changing button should show clear status after completion.
- The UI should tolerate empty states gracefully, especially before dataset upload, before baseline creation, and before any live session images arrive.
