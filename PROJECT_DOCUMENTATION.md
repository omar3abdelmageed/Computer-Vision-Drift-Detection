# Project Documentation

This document explains the project in simple plain text. It excludes generated folders and files such as `venv/`, `__pycache__/`, `.git/`, and `*.pyc`.

## Numbering System

- Folders use numbers like `1`, `2`, `3`.
- Files use numbers like `1.1`, `2.4`, `3.2`.
- Functions use numbers like `1.1.F1`.
- Classes, enums, and dataclasses use numbers like `3.1.C1`.
- When one function calls another project function, the called function is referenced by its number.

## Folder Structure

```text
CV_V2/
├── streamlit_app.py
├── README.md
├── requirements.txt
├── PROJECT_DOCUMENTATION.md
├── app/
│   ├── __init__.py
│   ├── components/
│   │   ├── __init__.py
│   │   ├── image_viewer.py
│   │   └── status_cards.py
│   └── tabs/
│       ├── __init__.py
│       ├── live_sessions.py
│       └── model_registration.py
├── backend/
│   ├── __init__.py
│   ├── common.py
│   ├── alerts/
│   │   ├── __init__.py
│   │   └── alert_rules.py
│   ├── baseline/
│   │   ├── __init__.py
│   │   ├── build_baseline.py
│   │   └── persistence.py
│   ├── database/
│   │   ├── __init__.py
│   │   ├── repositories.py
│   │   └── supabase_client.py
│   ├── drift/
│   │   ├── __init__.py
│   │   └── industrial_metrics.py
│   ├── features/
│   │   ├── __init__.py
│   │   └── image_properties.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── classification_validator.py
│   │   ├── dataset_validation.py
│   │   ├── detection_validator.py
│   │   ├── image_scanner.py
│   │   └── yaml_discovery.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── compatibility.py
│   │   ├── inference.py
│   │   └── yolo.py
│   ├── monitoring/
│   │   ├── __init__.py
│   │   └── session_manager.py
│   ├── registration/
│   │   └── local_files.py
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       ├── hashing.py
│       └── serialization.py
├── secrets/
│   └── .gitkeep
├── supabase/
│   └── schema.sql
└── tests/
    └── test_core.py
```

## Startup Flow

1. The app starts with `python -m streamlit run streamlit_app.py`.
2. Streamlit executes `streamlit_app.py` from top to bottom.
3. `1.1.F4 main()` sets Streamlit page settings.
4. `1.1.F4 main()` checks whether `auth_user` exists in Streamlit session state.
5. If no user is signed in, `1.1.F4 main()` calls `1.1.F2 render_auth_screen()`.
6. `1.1.F2 render_auth_screen()` calls `1.1.F1 supabase_is_configured()` to confirm required Supabase credentials exist.
7. `1.1.F1 supabase_is_configured()` calls `3.11.F1 AppConfig.from_env()` to load environment settings from `secrets/.env`.
8. If credentials are missing, Streamlit shows setup instructions and stops.
9. If credentials exist, `1.1.F2 render_auth_screen()` calls `3.4.F1 create_supabase_client()` with the anonymous key and shows sign in/sign up forms.
10. After successful sign in, the user and session are saved in Streamlit session state and Streamlit reruns.
11. On the authenticated rerun, `1.1.F4 main()` calls `1.1.F3 render_authenticated_header()`.
12. `1.1.F4 main()` calls `3.4.F1 create_supabase_client()` with service-role access for database persistence.
13. `1.1.F4 main()` creates two main tabs.
14. In the first tab it calls `2.2.F1 render_model_registration_tab()`.
15. In the second tab it calls `2.3.F1 render_live_sessions_tab()`.

## 1. Root Files

The root contains the app entry point, setup notes, dependency list, and this documentation.

### 1.1 `streamlit_app.py`

This is the Streamlit entry point. It controls app startup, authentication, Supabase client creation, and the two main tabs.

#### 1.1.F1 `supabase_is_configured()`

Checks whether Supabase credentials are available.

What it does:

1. Loads app configuration from environment variables.
2. Returns `True` only when both `SUPABASE_URL` and `SUPABASE_ANON_KEY` exist.

Calls: `3.11.F1 AppConfig.from_env()`.

#### 1.1.F2 `render_auth_screen()`

Shows the unauthenticated screen.

What it does:

1. Shows the app title.
2. Calls `1.1.F1 supabase_is_configured()`.
3. If Supabase is not configured, shows instructions and stops the Streamlit run.
4. Creates an anonymous Supabase client.
5. Shows sign in and create account tabs.
6. On successful sign in or sign up, stores the auth data in Streamlit session state.
7. Reruns Streamlit so the authenticated app can load.

Calls: `1.1.F1 supabase_is_configured()`, `3.4.F1 create_supabase_client()`.

#### 1.1.F3 `render_authenticated_header()`

Shows the header for a signed-in user.

What it does:

1. Reads the signed-in user from Streamlit session state.
2. Shows the app title and user email.
3. Provides a sign out button.
4. On sign out, signs out from Supabase, clears session state, and reruns the app.

Calls: `3.4.F1 create_supabase_client()`.

#### 1.1.F4 `main()`

Runs the main Streamlit app.

What it does:

1. Sets the page title and wide layout.
2. Sends unauthenticated users to `1.1.F2 render_auth_screen()`.
3. Shows the authenticated header.
4. Creates the service-role Supabase client.
5. Stops if persistence is unavailable.
6. Creates the `Model Management` and `Live Sessions` tabs.
7. Renders the model registration UI.
8. Renders the live monitoring UI.

Calls: `1.1.F2 render_auth_screen()`, `1.1.F3 render_authenticated_header()`, `3.4.F1 create_supabase_client()`, `2.2.F1 render_model_registration_tab()`, `2.3.F1 render_live_sessions_tab()`.

### 1.2 `README.md`

Explains the purpose of the project, setup steps, how to run the app, the expected workflow, and the drift tests implemented by the backend.

### 1.3 `requirements.txt`

Lists Python dependencies: Streamlit, Supabase client, dotenv support, Ultralytics YOLO, Pillow, NumPy, pandas, SciPy, PyYAML, Altair, pytest, and pytest-cov.

### 1.4 `PROJECT_DOCUMENTATION.md`

This file. It documents the folder structure, files, functions, classes, and important call relationships.

## 2. `app/`

The `app/` folder contains Streamlit UI code. It does not own the business logic; it calls backend functions and repositories.

### 2.1 `app/__init__.py`

Marks `app/` as a Python package.

### 2.2 `app/tabs/model_registration.py`

Builds the `Model Management` tab. This tab lets the user create/select a model, register dataset and model artifact paths, validate data, check model compatibility, build baselines, and view baseline records.

#### 2.2.F1 `render_model_registration_tab(client)`

Coordinates the full model registration workflow.

What it does:

1. Creates table repositories for models, datasets, dataset paths, artifacts, and baselines.
2. Lists models from Supabase.
3. Lets the user select or create a model.
4. Renders task type, dataset registration, artifact registration, validation, compatibility, baseline building, and baseline report sections.

Calls: `3.4.C1 SupabaseRepository`, `2.2.F2 select_or_create_model()`, `2.2.F7 render_task_type()`, `2.2.F8 render_dataset_registration()`, `2.2.F9 render_model_registration()`, `2.2.F10 render_dataset_validation()`, `2.2.F12 render_compatibility()`, `2.2.F14 render_baseline_builder()`, `2.2.F15 render_baseline_report()`.

#### 2.2.F2 `select_or_create_model(models, all_models)`

Shows the model selector.

What it does:

1. Builds a selectbox with `Add new model` and existing models.
2. Stores the selected model ID in Streamlit session state.
3. If creating a new model, renders the create form.
4. If selecting an existing model, renders its metadata editor.

Calls: `2.2.F3 format_model_option()`, `2.2.F4 render_create_model_form()`, `2.2.F5 render_selected_model_metadata()`.

#### 2.2.F3 `format_model_option(model)`

Creates the label shown in the model selectbox.

What it does:

1. Reads model name, status, and ID.
2. Returns a compact label with the first part of the ID.

Calls: none.

#### 2.2.F4 `render_create_model_form(models)`

Shows the form for creating a new model registration.

What it does:

1. Collects name, description, and tags.
2. Inserts a model row with default task type `object_detection`.
3. Stores the new model ID in Streamlit session state.
4. Reruns the app.

Calls: none.

#### 2.2.F5 `render_selected_model_metadata(models, selected)`

Shows information and edit controls for the selected model.

What it does:

1. Shows model name and status.
2. Provides a delete button.
3. Shows technical details in an expander.
4. Provides metadata edit fields.
5. Updates model metadata and reruns after saving.

Calls: `2.2.F6 render_delete_model_dialog()`.

#### 2.2.F6 `render_delete_model_dialog(models, selected)`

Shows a confirmation dialog before deleting a model.

What it does:

1. Warns that deleting the model removes related data.
2. Lets the user cancel.
3. Deletes the selected model if confirmed.
4. Resets selected model state and reruns.

Calls: none.

#### 2.2.F7 `render_task_type(models, selected_model)`

Lets the user choose object detection or classification.

What it does:

1. Shows a radio button for the task type.
2. If the task changes, resets validation, baseline, and registration status fields.
3. Shows detected task and validation status metrics.
4. Returns the selected task type.

Calls: none.

#### 2.2.F8 `render_dataset_registration(datasets, models, model_id, task_type)`

Lets the user register a local YOLO dataset directory.

What it does:

1. Loads the latest dataset row for the model.
2. Shows a dataset path form.
3. Validates the local folder path.
4. Inserts a dataset row.
5. Updates the model registration status.
6. Shows a dataset status card.

Calls: `3.10.F2 resolve_dataset_root()`, `2.4.F1 status_card()`.

#### 2.2.F9 `render_model_registration(artifacts, models, model_id)`

Lets the user register a local YOLO `.pt` artifact.

What it does:

1. Loads the latest artifact row for the model.
2. Shows a model artifact path form.
3. Validates the local artifact path.
4. Loads metadata from the YOLO artifact.
5. Inserts a model artifact row.
6. Updates the model registration status.
7. Shows artifact status and raw metadata.

Calls: `3.10.F3 resolve_model_artifact()`, `3.8.3.F1 load_yolo_metadata()`, `2.4.F1 status_card()`.

#### 2.2.F10 `render_dataset_validation(datasets, dataset_paths, models, dataset_record, task_type)`

Validates the registered dataset.

What it does:

1. Stops with an info message if no dataset is registered.
2. Runs validation when the refresh button is clicked.
3. Also runs a validation preview on normal render.
4. Persists validation results when they differ from the database row.
5. Shows validation status, split counts, and issues.

Calls: `3.6.F1 validate_dataset()`, `2.2.F11 persist_dataset_validation()`, `2.4.F1 status_card()`, `3.13.F1 to_jsonable()`.

#### 2.2.F11 `persist_dataset_validation(datasets, dataset_paths, models, dataset_record, metadata)`

Saves dataset validation metadata to Supabase.

What it does:

1. Converts validation metadata into database columns.
2. Updates the dataset row when values changed.
3. Upserts per-split dataset path rows.
4. Updates model task validation and registration status.
5. Returns `True` when it changed persisted data.

Calls: `3.13.F1 to_jsonable()`.

#### 2.2.F12 `render_compatibility(artifacts, models, artifact_record, dataset_metadata)`

Checks whether the registered model artifact matches the dataset.

What it does:

1. Stops with an info message if artifact or dataset metadata is missing.
2. Builds a small artifact metadata object from the database row.
3. Runs compatibility checking.
4. Persists the result if it changed.
5. Shows compatibility status and details.
6. Returns artifact metadata only when compatible.

Calls: `3.8.1.F1 check_compatibility()`, `2.2.F13 persist_compatibility_result()`, `2.4.F1 status_card()`.

#### 2.2.F13 `persist_compatibility_result(artifacts, models, artifact_record, result)`

Saves model compatibility results.

What it does:

1. Compares the new compatibility result to the artifact row.
2. Updates the artifact row if needed.
3. Updates the model registration status to `compatible` or `failed`.
4. Returns `True` when it changed persisted data.

Calls: none.

#### 2.2.F14 `render_baseline_builder(models, baselines, selected_model, dataset_record, artifact_record, dataset_metadata, artifact_metadata)`

Builds a baseline profile for a compatible model and dataset.

What it does:

1. Enables the button only when dataset, artifact, validation, and compatibility are ready.
2. Marks baseline status as running.
3. Builds the baseline profile from train and validation data.
4. Saves baseline sections to Supabase.
5. Marks the model as completed or failed.
6. Shows the current baseline status.

Calls: `3.2.F1 build_baseline_profile()`, `3.3.F1 save_baseline_profiles()`, `2.4.F1 status_card()`.

#### 2.2.F15 `render_baseline_report(baselines, model_id)`

Shows stored baseline profile rows.

What it does:

1. Loads baseline rows for the selected model.
2. Shows an info message if no baselines exist.
3. Shows rows in a dataframe.
4. Shows raw baseline JSON in an expander.

Calls: none.

### 2.3 `app/tabs/live_sessions.py`

Builds the `Live Sessions` tab. This tab lets the user select a baseline-ready model, configure an image source, start or stop monitoring sessions, review predictions, save feedback, and view drift charts.

#### 2.3.F1 `render_live_sessions_tab(client)`

Coordinates the full live session workflow.

What it does:

1. Creates repositories for models, datasets, artifacts, sources, sessions, images, predictions, drift, feedback, and baselines.
2. Lets the user choose a registered model.
3. Finds the latest source and latest monitoring session.
4. Runs automatic polling if conditions are met.
5. Renders source, session, prediction, and drift subtabs.

Calls: `3.4.C1 SupabaseRepository`, `2.3.F2 select_registered_model()`, `2.3.F4 latest_source()`, `2.3.F5 latest_monitoring_session()`, `2.3.F6 source_for_session()`, `2.3.F9 maybe_auto_poll()`, `2.3.F7 render_source_tab()`, `2.3.F8 render_session_tab()`, `2.3.F15 render_predictions_tab()`, `2.3.F24 render_drift_tab()`.

#### 2.3.F2 `select_registered_model(models, datasets, artifacts, baselines)`

Lets the user choose a model for live monitoring.

What it does:

1. Lists models from Supabase.
2. Shows a selectbox of registered models.
3. Loads the latest dataset and artifact for the selected model.
4. Loads baseline profiles for the current dataset and artifact.
5. Calculates whether the model is eligible for live monitoring.
6. Shows an eligibility status card.

Calls: `2.3.F3 current_baseline_profiles()`, `2.4.F1 status_card()`.

#### 2.3.F3 `current_baseline_profiles(baselines, model_id, dataset, artifact)`

Finds baseline profiles matching the selected model, dataset, and artifact.

Calls: none.

#### 2.3.F4 `latest_source(sources, model_id)`

Returns the newest production source for a model.

Calls: none.

#### 2.3.F5 `latest_monitoring_session(sessions, model_id)`

Returns the newest monitoring session for a model.

Calls: none.

#### 2.3.F6 `source_for_session(sources, fallback_source, session)`

Chooses which source should be used for the current session.

What it does:

1. If no running or paused session exists, returns the latest source.
2. If a running or paused session exists, returns the source tied to that session.
3. Falls back to the latest source if the session has no source ID.

Calls: none.

#### 2.3.F7 `render_source_tab(sources, model_id, source, protected_source_id)`

Shows and saves the image source configuration.

What it does:

1. Shows fields for watched folder path and polling interval.
2. Inserts or updates a `production_sources` row.
3. Avoids overwriting a source currently protected by a running or paused session.
4. Shows source status based on whether the folder exists.

Calls: `2.4.F1 status_card()`.

#### 2.3.F8 `render_session_tab(client, sessions, images, predictions, feedback, model, source, latest_session)`

Shows session controls.

What it does:

1. Builds a default session name.
2. Enables start only when the model is eligible, source exists, artifact path exists, and no active latest session exists.
3. Inserts a session row when starting.
4. Processes images once immediately after start.
5. Supports pause, resume, and stop.
6. Builds and saves a summary when stopping.
7. Shows session status and summary.

Calls: `2.3.F10 process_source_once_and_report()`, `2.3.F12 build_session_summary()`, `2.4.F1 status_card()`.

#### 2.3.F9 `maybe_auto_poll(client, model, source, session)`

Registers a Streamlit fragment that periodically processes new source images.

What it does:

1. Returns early unless there is a running session, eligible model, valid source, and artifact path.
2. Uses source polling interval.
3. Uses Streamlit `st.fragment` when available.
4. Tracks last poll time in Streamlit session state.
5. Calls processing when the interval has elapsed.

Calls: `2.3.F10 process_source_once_and_report()`.

#### 2.3.F10 `process_source_once_and_report(client, model, source, session, show_empty)`

Runs one processing pass and reports the result in the UI.

What it does:

1. Reads artifact and baseline profile data from the selected model object.
2. Calls backend source processing.
3. Catches processing errors and shows them.
4. Shows a user-friendly summary.

Calls: `3.15.F1 process_source_once()`, `2.3.F11 report_processing_summary()`.

#### 2.3.F11 `report_processing_summary(summary, show_empty)`

Shows the result of processing a source folder.

What it does:

1. Shows success when images were processed.
2. Shows warning when the source path is missing.
3. Shows info when no new images were found.

Calls: none.

#### 2.3.F12 `build_session_summary(images, predictions, feedback, session, source, ended_at)`

Builds a summary for a completed monitoring session.

What it does:

1. Loads images for the session.
2. Loads predictions for those images.
3. Loads latest feedback for those predictions.
4. Calculates processed image count, prediction count, average confidence, review counts, and duration.

Calls: `2.3.F13 predictions_for_images()`, `2.3.F14 latest_feedback_for_predictions()`, `2.3.F23 duration_seconds()`.

#### 2.3.F13 `predictions_for_images(predictions, image_rows)`

Collects all prediction rows for a list of images.

Calls: none.

#### 2.3.F14 `latest_feedback_for_predictions(feedback, prediction_rows)`

Collects the newest feedback row for each prediction.

Calls: none.

#### 2.3.F15 `render_predictions_tab(client, model, images, predictions, feedback, session)`

Shows predictions for the latest session.

What it does:

1. Stops with an info message if there is no session.
2. Loads recent images for the session.
3. Shows an info message if no images were processed.
4. Renders each image review group.

Calls: `2.3.F16 render_image_review_group()`.

#### 2.3.F16 `render_image_review_group(client, model, session, image, prediction_rows, feedback)`

Shows one image and its predictions.

What it does:

1. Shows the image preview.
2. Draws detection overlays when detection predictions exist.
3. Shows classification or detection prediction details.
4. Shows feedback controls when predictions exist.

Calls: `2.5.F3 draw_detection_overlays()`, `2.3.F17 detection_summary_rows()`, `2.3.F19 render_feedback_table()`.

#### 2.3.F17 `detection_summary_rows(prediction_rows)`

Creates rows for a detection prediction dataframe.

Calls: `2.3.F18 normalized_box_label()`.

#### 2.3.F18 `normalized_box_label(prediction)`

Formats normalized detection box values as text.

Calls: none.

#### 2.3.F19 `render_feedback_table(client, model, session, image, prediction_rows, feedback)`

Shows the feedback table for an image.

Calls: `2.3.F20 render_prediction_feedback_form()`.

#### 2.3.F20 `render_prediction_feedback_form(client, model, session, image, prediction, feedback, index)`

Shows and saves feedback for a single prediction.

What it does:

1. Loads latest feedback for the prediction.
2. Chooses feedback options based on task type.
3. Lets the user approve, reject, or correct the prediction.
4. Validates corrected boxes.
5. Inserts a feedback row.
6. Recalculates feedback-based concept drift.

Calls: `2.3.F18 normalized_box_label()`, `2.3.F21 parse_corrected_box()`, `3.15.F11 record_feedback_concept_drift()`.

#### 2.3.F21 `parse_corrected_box(value)`

Parses a corrected detection box from `x,y,w,h` text.

What it does:

1. Converts comma-separated values to floats.
2. Requires exactly four values.
3. Requires every value to be between 0 and 1.
4. Returns an empty dict for invalid input.

Calls: none.

#### 2.3.F22 `parse_datetime(value)`

Converts an ISO date string to a timezone-aware datetime.

Calls: none.

#### 2.3.F23 `duration_seconds(started_at, ended_at)`

Calculates session duration in seconds.

Calls: `2.3.F22 parse_datetime()`.

#### 2.3.F24 `render_drift_tab(drift, alerts, session)`

Shows an alert-first drift overview, status metrics, and diagnostic charts for the latest session.

What it does:

1. Stops with an info message if there is no session.
2. Loads drift rows for the session.
3. Adds normalized drift scores.
4. Loads unresolved drift alerts linked to drift result rows.
5. Shows a drift overview with session status, the fixed latest-25-image window, latest check time, and overall alert state.
6. Shows active alerts before charts, grouped by severity.
7. Shows status-first top-level metrics.
8. Renders total data drift, property drift, prediction distribution, and concept summary sections.

Calls: `2.3.F25 with_drift_score()`, `2.3.F29 latest_metric_row()`, `2.3.F27 format_score()`, `2.3.F28 format_percent()`, `2.3.F31 render_total_data_drift_chart()`, `2.3.F32 render_data_property_charts()`, `2.3.F35 render_prediction_distribution_chart()`, `2.3.F37 render_concept_summary()`.

#### 2.3.F25 `with_drift_score(row)`

Copies a drift row and adds a numeric score.

Calls: `2.3.F26 normalized_drift_score()`.

#### 2.3.F26 `normalized_drift_score(row)`

Converts a drift row into a numeric score.

What it does:

1. Uses `metric_value` if it is numeric.
2. Otherwise maps statuses to `0`, `60`, or `100`.

Calls: none.

#### 2.3.F27 `format_score(score)`

Formats drift score text for UI metrics.

Calls: none.

#### 2.3.F28 `format_percent(value)`

Formats percentage text for UI metrics.

Calls: none.

#### 2.3.F29 `latest_metric_row(rows, metric_name)`

Finds the latest drift row for one metric name.

Calls: none.

#### 2.3.F30 `metric_rows(rows, metric_name)`

Finds all drift rows for one metric name, sorted by creation time.

Calls: none.

#### 2.3.F31 `render_total_data_drift_chart(rows)`

Shows the total data drift line chart with warning and critical threshold reference lines.

Calls: `2.3.F39 graph_heading()`, `2.3.F30 metric_rows()`, `2.3.F34 padded_time_axis()`.

#### 2.3.F32 `render_data_property_charts(rows)`

Shows brightness, contrast, colour distribution, and noise charts with latest status, percent-difference context, and window/threshold captions.

Calls: `2.3.F39 graph_heading()`, `2.3.F30 metric_rows()`, `2.3.F33 property_line_frame()`, `2.3.F34 padded_time_axis()`.

#### 2.3.F33 `property_line_frame(rows)`

Builds a pandas dataframe comparing training and production property averages.

Calls: none.

#### 2.3.F34 `padded_time_axis(frame)`

Builds an Altair time axis with padding around the first and last timestamp.

Calls: none.

#### 2.3.F35 `render_prediction_distribution_chart(row)`

Shows a bar chart comparing training and production class percentages.

Calls: `2.3.F39 graph_heading()`, `2.3.F36 class_distribution_frame()`.

#### 2.3.F36 `class_distribution_frame(training_percentages, production_percentages)`

Builds a pandas dataframe for prediction distribution charting.

Calls: none.

#### 2.3.F37 `render_concept_summary(row)`

Shows human review concept drift summary.

Calls: `2.3.F39 graph_heading()`, `2.3.F28 format_percent()`.

#### 2.3.F38 `drift_scores(rows)`

Builds a dictionary of the highest score per drift type.

Calls: none. This helper is present but not used by the current UI flow.

#### 2.3.F39 `graph_heading(title, help_text, key)`

Shows a chart heading with a disabled help button containing explanation text.

Calls: none.

### 2.4 `app/components/status_cards.py`

Contains small reusable Streamlit status UI.

#### 2.4.F1 `status_card(title, status, detail)`

Shows a compact status block.

What it does:

1. Displays the title in bold.
2. Displays the status as inline code.
3. Displays optional detail text.

Calls: none.

### 2.5 `app/components/image_viewer.py`

Contains image preview helpers and detection overlay drawing.

#### 2.5.F1 `render_image(path, caption)`

Shows an image in Streamlit when a path exists.

Calls: none.

#### 2.5.F2 `yolo_box_to_pixel_rect(prediction, image_width, image_height)`

Converts a normalized YOLO detection box to pixel rectangle coordinates.

What it does:

1. Reads `x_center`, `y_center`, `width`, and `height`.
2. Rejects missing or non-numeric values.
3. Converts normalized values to pixel values.
4. Clips the result to the image size.

Calls: none.

#### 2.5.F3 `draw_detection_overlays(path, predictions)`

Draws detection boxes and labels on an image.

What it does:

1. Opens the image.
2. Converts predictions to pixel rectangles.
3. Adds padding to each rectangle.
4. Draws box corners and class labels.
5. Returns the annotated image.

Calls: `2.5.F2 yolo_box_to_pixel_rect()`, `2.5.F4 padded_rect()`, `2.5.F5 draw_box_corners()`.

#### 2.5.F4 `padded_rect(rect, image_width, image_height)`

Expands a rectangle slightly while staying inside image bounds.

Calls: none.

#### 2.5.F5 `draw_box_corners(draw, rect, color, line_width)`

Draws corner-style detection box lines.

Calls: none.

### 2.6 `app/components/__init__.py`

Marks `app/components/` as a Python package.

### 2.7 `app/tabs/__init__.py`

Marks `app/tabs/` as a Python package.

## 3. `backend/`

The `backend/` folder contains business logic: shared data structures, database access, dataset validation, YOLO metadata and inference, baseline building, drift calculation, monitoring sessions, and utility functions.

### 3.1 `backend/common.py`

Defines shared enums and dataclasses used across the backend and UI.

Classes and data structures:

- 3.1.C1 `TaskType`: user-selected model task, either object detection or classification.
- 3.1.C2 `DetectedTaskType`: task detected from dataset structure.
- 3.1.C3 `DatasetLayout`: dataset layout shape, such as YOLO detection or classification folders.
- 3.1.C4 `SplitName`: dataset split names: train, val, and test.
- 3.1.C5 `ValidationStatus`: valid, warning, invalid, or pending.
- 3.1.C6 `BaselineStatus`: baseline lifecycle states.
- 3.1.C7 `SourceType`: image source kinds.
- 3.1.C8 `SourceMode`: source mode, such as baseline or live monitoring.
- 3.1.C9 `SessionStatus`: monitoring session lifecycle states.
- 3.1.C10 `DriftType`: kinds of drift measured by the app.
- 3.1.C11 `DriftStatus`: ok, warning, or critical.
- 3.1.C12 `FeedbackType`: human feedback categories.
- 3.1.C13 `ValidationIssue`: a validation problem with severity, code, message, path, and details.
- 3.1.C14 `YamlCandidate`: a discovered YOLO YAML file and its parsed metadata.
- 3.1.C15 `DatasetSplitPaths`: paths and counts for one dataset split.
- 3.1.C16 `ClassMapping`: class ID and class name.
- 3.1.C17 `DatasetMetadata`: complete validated dataset metadata.
- 3.1.C18 `ImageMetadata`: image file metadata.
- 3.1.C19 `ImageFeatures`: extracted image quality and color features.
- 3.1.C20 `DetectionLabel`: one parsed YOLO detection label.
- 3.1.C21 `ClassificationLabel`: one classification label.
- 3.1.C22 `DriftResult`: one calculated drift metric result.

### 3.2 `backend/baseline/build_baseline.py`

Builds baseline profiles from training and validation data.

#### 3.2.F1 `build_baseline_profile(dataset, artifact_path, task_type)`

Builds the full baseline profile.

What it does:

1. Scans train and validation image folders.
2. Extracts image metadata and image features.
3. Keeps training feature rows separate for reference metrics.
4. Optionally runs YOLO inference on baseline images.
5. Returns dataset profile, image stats, feature rows, class distribution, and prediction summary.

Calls: `3.2.F2 split_for_image_path()`, `3.7.F2 scan_images()`, `3.12.F2 extract_image_metadata()`, `3.12.F3 extract_image_features()`, `3.8.2.F1 run_yolo_inference()`, `3.13.F1 to_jsonable()`, `3.2.F3 summarize_features()`, `3.5.F5 property_averages()`, `3.5.F6 feature_vectors()`, `3.2.F4 summarize_class_distribution()`, `3.2.F9 summarize_predictions()`.

#### 3.2.F2 `split_for_image_path(dataset, image_path)`

Finds whether an image belongs to the train, val, or test split.

Calls: none.

#### 3.2.F3 `summarize_features(feature_rows)`

Summarizes numeric feature rows.

What it does:

1. Returns count zero when no feature rows exist.
2. Calculates averages for numeric feature keys.

Calls: none.

#### 3.2.F4 `summarize_class_distribution(dataset)`

Builds class distribution summaries for the baseline.

Calls: `3.2.F5 training_class_counts()`, `3.2.F8 class_distribution_percentages()`.

#### 3.2.F5 `training_class_counts(dataset)`

Counts training classes based on dataset task type.

Calls: `3.2.F6 classification_training_counts()`, `3.2.F7 detection_training_counts()`.

#### 3.2.F6 `classification_training_counts(images_path)`

Counts classification training images by class folder name.

Calls: none.

#### 3.2.F7 `detection_training_counts(labels_path, class_names)`

Counts object detection labels by class name.

Calls: none.

#### 3.2.F8 `class_distribution_percentages(counts)`

Converts class counts to percentages.

Calls: none.

#### 3.2.F9 `summarize_predictions(prediction_rows)`

Summarizes model predictions made during baseline building.

Calls: `3.5.F7 class_distribution()`, `3.5.F8 class_percentages()`.

### 3.3 `backend/baseline/persistence.py`

Saves baseline sections into Supabase rows.

#### 3.3.F1 `save_baseline_profiles(baselines, model_id, dataset_id, artifact_id, profile)`

Persists baseline profile sections.

What it does:

1. Iterates through named profile sections.
2. Builds a row for each section.
3. Updates existing rows for the same model, dataset, artifact, and profile type.
4. Inserts missing rows.

Calls: none.

### 3.4 `backend/database/`

Contains Supabase client creation and a small repository wrapper.

#### 3.4.F1 `create_supabase_client(use_service_role)`

Creates a Supabase client from environment configuration.

What it does:

1. Loads app configuration.
2. Chooses service-role key when requested and available.
3. Falls back to anon key otherwise.
4. Returns `None` if URL or key is missing.
5. Imports Supabase and creates the client.

Calls: `3.11.F1 AppConfig.from_env()`.

#### 3.4.C1 `SupabaseRepository`

A small wrapper around one Supabase table.

Methods:

1. `list()`: selects all rows ordered by a timestamp column.
2. `get()`: gets one row by ID.
3. `insert()`: inserts one row.
4. `update()`: updates one row by ID.
5. `delete()`: deletes one row by ID.
6. `upsert()`: upserts a row, optionally with conflict keys.
7. `where()`: selects rows matching filters.
8. `where_ordered()`: selects filtered rows with ordering and optional limit.
9. `latest_where()`: returns the newest matching row.

Calls: repository methods call Supabase SDK query methods.

### 3.5 `backend/drift/industrial_metrics.py`

Contains numeric helpers for image property drift, feature vector drift, prediction distribution drift, and review summary metrics.

#### 3.5.F1 `colorfulness_from_rgb_channel_stats(rgb_means, rgb_stds)`

Calculates a colorfulness score from RGB channel statistics.

Calls: none.

#### 3.5.F2 `feature_vector(row)`

Converts one feature row into a fixed numeric vector.

Calls: none.

#### 3.5.F3 `property_value(property_key, row)`

Extracts one named image property value from a feature row.

Calls: `3.5.F1 colorfulness_from_rgb_channel_stats()`.

#### 3.5.F4 `mean(values)`

Returns the average of numeric values, or `None` when no values exist.

Calls: none.

#### 3.5.F5 `property_averages(feature_rows)`

Calculates average brightness, contrast, colour distribution, and noise.

Calls: `3.5.F3 property_value()`, `3.5.F4 mean()`.

#### 3.5.F6 `feature_vectors(feature_rows)`

Converts many feature rows into numeric vectors.

Calls: `3.5.F2 feature_vector()`.

#### 3.5.F7 `class_distribution(labels)`

Counts labels by class name.

Calls: none.

#### 3.5.F8 `class_percentages(counts)`

Converts class counts to percentages.

Calls: none.

#### 3.5.F9 `rbf_mmd(reference_vectors, current_vectors)`

Calculates RBF-kernel Maximum Mean Discrepancy between reference and current vectors.

Calls: `3.5.F10 rbf_kernel_mean()`.

#### 3.5.F10 `rbf_kernel_mean(left, right, gamma)`

Calculates the mean RBF kernel similarity between two vector sets.

Calls: none.

#### 3.5.F11 `total_variation_distance(reference_counts, current_counts)`

Calculates total variation distance between two class count dictionaries.

Calls: `3.5.F8 class_percentages()`.

#### 3.5.F12 `positive_int_counts(counts)`

Filters a class count dictionary down to positive integer counts.

Calls: none.

#### 3.5.F13 `chi_square_prediction_drift(training_counts, production_counts)`

Calculates prediction drift with a Chi-Square goodness-of-fit test against training class counts.

Calls: `3.5.F12 positive_int_counts()`.

#### 3.5.F14 `review_summary(feedback_types)`

Summarizes human feedback into review counts and accuracy percentage.

Calls: none.

### 3.6 `backend/ingestion/dataset_validation.py`

Routes dataset validation to the correct validator.

#### 3.6.F1 `validate_dataset(dataset_root, selected_task_type)`

Validates a dataset based on selected task type.

What it does:

1. Converts the selected task value to `TaskType`.
2. Uses detection validation for object detection.
3. Uses classification validation for classification.

Calls: `3.6.F2 validate_detection_dataset()`, `3.6.F8 validate_classification_dataset()`.

### 3.7 `backend/ingestion/image_scanner.py`

Contains image file scanning and image format validation helpers.

#### 3.7.F1 `is_supported_image(path)`

Checks whether a path has a supported image extension.

Calls: none.

#### 3.7.F2 `scan_images(path)`

Finds supported image files under a directory.

Calls: `3.7.F1 is_supported_image()`.

#### 3.7.F3 `validate_image(path)`

Opens an image with Pillow to verify it is readable.

Calls: none.

### 3.6 Detection Validation Functions

These functions live in `backend/ingestion/detection_validator.py`.

#### 3.6.F2 `validate_detection_dataset(dataset_root)`

Validates a YOLO object detection dataset.

What it does:

1. Discovers YOLO YAML files.
2. Chooses classes from YAML when available.
3. Resolves train, val, and test split paths.
4. Counts images and labels.
5. Validates label files.
6. Counts unsupported files.
7. Returns dataset metadata.

Calls: `3.6.F14 discover_yamls()`, `3.6.F3 resolve_detection_layout()`, `3.6.F4 hydrate_split_counts()`, `3.6.F5 validate_detection_split()`, `3.6.F7 count_unsupported_files()`.

#### 3.6.F3 `resolve_detection_layout(root)`

Finds detection image and label folders for train, val, and test splits.

Calls: none.

#### 3.6.F4 `hydrate_split_counts(splits)`

Adds image and label counts to split path objects.

Calls: `3.7.F2 scan_images()`.

#### 3.6.F5 `validate_detection_split(split_paths, class_names)`

Validates labels for one detection split.

Calls: `3.6.F6 parse_detection_label_file()`.

#### 3.6.F6 `parse_detection_label_file(label_path, class_names)`

Parses one YOLO detection label file.

What it does:

1. Reads each line.
2. Requires five values per label.
3. Validates class ID.
4. Validates normalized box coordinates.
5. Returns parsed labels and validation issues.

Calls: none.

#### 3.6.F7 `count_unsupported_files(root)`

Counts files under a detection dataset that are not supported images, labels, YAML files, or common text metadata.

Calls: `3.7.F1 is_supported_image()`.

### 3.6 Classification Validation Functions

These functions live in `backend/ingestion/classification_validator.py`.

#### 3.6.F8 `validate_classification_dataset(dataset_root)`

Validates a YOLO classification dataset.

What it does:

1. Resolves split folders.
2. Collects class names from training folders.
3. Counts images in train, val, and test.
4. Checks class alignment across splits.
5. Counts unsupported files.
6. Returns dataset metadata.

Calls: `3.6.F9 resolve_classification_layout()`, `3.6.F10 validate_class_alignment()`, `3.6.F11 count_unsupported_files()`.

#### 3.6.F9 `resolve_classification_layout(root)`

Finds classification train, val, and test folders and class names.

Calls: `3.7.F2 scan_images()`.

#### 3.6.F10 `validate_class_alignment(train_path, other_path, issues)`

Checks that a validation or test split has the same class folders as train.

Calls: none.

#### 3.6.F11 `count_unsupported_files(root)`

Counts unsupported files in a classification dataset.

Calls: `3.7.F1 is_supported_image()`.

### 3.6 YAML Discovery Functions

These functions live in `backend/ingestion/yaml_discovery.py`.

#### 3.6.F14 `discover_yamls(dataset_root)`

Finds and parses YAML files under a dataset root.

What it does:

1. Recursively finds `.yaml` and `.yml` files.
2. Parses each file.
3. Extracts class names.
4. Records parse issues when files cannot be parsed.

Calls: `3.6.F15 normalize_names()`.

#### 3.6.F15 `normalize_names(names)`

Normalizes YAML class names from list or dictionary shape into a list of strings.

Calls: none.

### 3.8 `backend/model/`

The `backend/model/` folder contains YOLO artifact metadata loading, compatibility checks, and inference parsing.

### 3.8.1 `backend/model/compatibility.py`

Checks whether a YOLO artifact matches a dataset.

#### 3.8.1.F1 `check_compatibility(dataset, artifact)`

Compares dataset metadata with model artifact metadata.

What it does:

1. Checks task type match.
2. Checks class count match.
3. Checks class name match.
4. Returns compatible status, validation status, and issue details.

Calls: none.

### 3.8.2 `backend/model/inference.py`

Runs YOLO inference and normalizes prediction results.

#### 3.8.2.F1 `run_yolo_inference(model_path, image_path, task_type)`

Runs YOLO on one image.

What it does:

1. Imports Ultralytics YOLO lazily.
2. Loads the model artifact.
3. Runs inference on the image.
4. Measures inference time.
5. Parses classification results when task type is classification.
6. Parses detection results otherwise.

Calls: `3.8.2.F2 parse_classification_results()`, `3.8.2.F3 parse_detection_results()`.

#### 3.8.2.F2 `parse_classification_results(results, inference_ms)`

Converts Ultralytics classification results into database-ready dictionaries.

What it does:

1. Reads class names and probability data.
2. Extracts top-1 class and confidence.
3. Stores top-k predictions.
4. Adds inference time.

Calls: none.

#### 3.8.2.F3 `parse_detection_results(results, inference_ms)`

Converts Ultralytics detection results into database-ready dictionaries.

What it does:

1. Reads boxes from each result.
2. Extracts predicted class and confidence.
3. Stores normalized box center, width, and height.
4. Adds inference time and raw prediction data.

Calls: none.

### 3.8.3 `backend/model/yolo.py`

Loads YOLO artifact metadata without running inference.

#### 3.8.3.C1 `ModelArtifactMetadata`

Dataclass for artifact path, artifact name, YOLO task, class names, class count, input size, and raw metadata.

#### 3.8.3.F1 `load_yolo_metadata(model_path)`

Loads metadata from a YOLO `.pt` artifact.

What it does:

1. Imports Ultralytics YOLO lazily.
2. Loads the model file.
3. Reads class names.
4. Reads task type.
5. Reads input size from model overrides.
6. Returns `3.8.3.C1 ModelArtifactMetadata`.

Calls: none.

### 3.10 `backend/registration/local_files.py`

Validates local dataset and model artifact paths entered by the user.

#### 3.10.F1 `resolve_local_path(raw_path)`

Expands and resolves a user-entered local path.

Calls: none.

#### 3.10.F2 `resolve_dataset_root(raw_path)`

Validates that a path is an existing directory.

Calls: `3.10.F1 resolve_local_path()`.

#### 3.10.F3 `resolve_model_artifact(raw_path)`

Validates that a path is an existing `.pt` file.

Calls: `3.10.F1 resolve_local_path()`.

### 3.11 `backend/utils/config.py`

Loads configuration from `secrets/.env` and environment variables.

#### 3.11.F2 `load_env_file(path)`

Loads a dotenv file if `python-dotenv` is installed.

Calls: none.

#### 3.11.C1 `AppConfig`

Dataclass for Supabase URL/key settings, local workspace folder, app environment, and log level.

#### 3.11.F1 `AppConfig.from_env()`

Builds an `AppConfig` from environment variables.

What it does:

1. Calls `3.11.F2 load_env_file()`.
2. Reads Supabase settings.
3. Reads local workspace directory.
4. Reads app environment and log level defaults.

Calls: `3.11.F2 load_env_file()`.

### 3.12 `backend/features/image_properties.py`

Extracts image metadata and image quality/color features.

#### 3.12.F1 `load_image_rgb(path)`

Opens an image and converts it to RGB.

Calls: none.

#### 3.12.F2 `extract_image_metadata(path, split)`

Reads metadata from one image.

What it does:

1. Opens the image.
2. Reads format, mode, dimensions, aspect ratio, file size, and bit depth.
3. Computes file hash.
4. Returns `3.1.C18 ImageMetadata`.

Calls: `3.14.F1 file_sha256()`.

#### 3.12.F3 `extract_image_features(path)`

Extracts numeric image features.

What it does:

1. Loads the image as RGB.
2. Calculates brightness and contrast.
3. Calculates sharpness and saturation.
4. Calculates edge density, RGB statistics, entropy, noise estimate, and color histogram.
5. Returns `3.1.C19 ImageFeatures`.

Calls: `3.12.F1 load_image_rgb()`.

### 3.13 `backend/utils/serialization.py`

Converts Python objects into JSON-safe values.

#### 3.13.F1 `to_jsonable(value)`

Converts dataclasses, enums, paths, dictionaries, lists, tuples, and primitive values into JSON-friendly data.

Calls: itself recursively.

### 3.14 `backend/utils/hashing.py`

Contains file hashing helpers.

#### 3.14.F1 `file_sha256(path, chunk_size)`

Computes the SHA-256 hash of a file in chunks.

Calls: none.

### 3.15 `backend/monitoring/session_manager.py`

Processes live session images, stores predictions, calculates drift, creates alerts, and records feedback-based concept drift.

#### 3.15.F1 `process_source_once(client, model, artifact, source, session, baseline_profiles, window_size)`

Processes new images from a source folder once.

What it does:

1. Creates repositories for images, predictions, drift, and alerts.
2. Checks that the source path exists.
3. Scans images.
4. Skips duplicate image content within the same session.
5. Extracts image metadata and features.
6. Inserts image rows.
7. Runs YOLO inference when an artifact path exists.
8. Inserts prediction rows.
9. Builds the current session window.
10. Calculates drift for the window.
11. Persists drift and alerts.

Calls: `3.4.C1 SupabaseRepository`, `3.7.F2 scan_images()`, `3.12.F2 extract_image_metadata()`, `3.12.F3 extract_image_features()`, `3.8.2.F1 run_yolo_inference()`, `3.13.F1 to_jsonable()`, `3.15.F2 normalize_image_source()`, `3.15.F3 current_session_window()`, `3.15.F5 calculate_drift_for_window()`, `3.15.F10 persist_drift_results()`.

#### 3.15.F2 `normalize_image_source(source_type)`

Converts source type names into the image source value stored in the database.

Calls: none.

#### 3.15.F3 `current_session_window(images_repo, predictions_repo, session_id, window_size)`

Loads the latest processed images and prediction labels for a session window.

Calls: `3.15.F4 feature_payload_from_image_row()`.

#### 3.15.F4 `feature_payload_from_image_row(row)`

Reads an image row's stored feature payload.

Calls: none.

#### 3.15.F5 `calculate_drift_for_window(baseline_profiles, current_features, current_predictions)`

Calculates drift metrics for the current live window.

What it does:

1. Reads baseline feature rows and training averages.
2. Calculates total data drift using MMD when enough vectors exist.
3. Calculates brightness, contrast, colour distribution, and noise drift.
4. Calculates Chi-Square prediction drift when predictions exist.
5. Returns drift result objects.

Calls: `3.5.F5 property_averages()`, `3.5.F6 feature_vectors()`, `3.5.F9 rbf_mmd()`, `3.5.F3 property_value()`, `3.15.F6 normalized_average_delta()`, `3.15.F7 status_from_score()`, `3.15.F8 status_from_mmd()`, `3.15.F9 status_from_chi_square()`, `3.5.F7 class_distribution()`, `3.5.F8 class_percentages()`, `3.5.F13 chi_square_prediction_drift()`.

#### 3.15.F6 `normalized_average_delta(training_average, production_average)`

Calculates relative difference between training and production averages.

Calls: none.

#### 3.15.F7 `status_from_score(score)`

Maps a general drift score to `ok`, `warning`, or `critical`.

Calls: none.

#### 3.15.F8 `status_from_mmd(score)`

Maps an MMD drift score to `ok`, `warning`, or `critical`.

Calls: none.

#### 3.15.F9 `status_from_chi_square(p_value, has_unseen_classes, has_training_counts)`

Maps Chi-Square prediction drift results to `ok`, `warning`, or `critical`.

Calls: none.

#### 3.15.F10 `persist_drift_results(drift_repo, alerts_repo, model_id, session_id, num_images, results)`

Saves drift results and alert rows.

What it does:

1. Inserts each drift result into Supabase.
2. Converts result data to JSON-safe values.
3. Creates alerts from non-ok drift results.
4. Inserts alerts.

Calls: `3.13.F1 to_jsonable()`, `3.16.F1 alerts_from_drift()`.

#### 3.15.F11 `record_feedback_concept_drift(client, model, session)`

Stores concept drift based on human review feedback.

What it does:

1. Returns early if there is no active session.
2. Loads feedback rows for the session.
3. Builds a review summary.
4. Inserts a concept drift row.

Calls: `3.4.C1 SupabaseRepository`, `3.5.F14 review_summary()`.

### 3.16 `backend/alerts/alert_rules.py`

Converts drift results into alert rows.

#### 3.16.F1 `alerts_from_drift(results)`

Creates alerts for drift results that are not OK.

What it does:

1. Skips OK results.
2. Marks critical results as critical and all other non-OK results as warning.
3. Builds title, message, severity, and details.

Calls: none.

### 3.17 `backend/__init__.py`

Marks `backend/` as a Python package.

### 3.18 Empty package marker files

The following files mark folders as Python packages and contain no project logic:

- `backend/alerts/__init__.py`
- `backend/baseline/__init__.py`
- `backend/database/__init__.py`
- `backend/drift/__init__.py`
- `backend/features/__init__.py`
- `backend/ingestion/__init__.py`
- `backend/model/__init__.py`
- `backend/monitoring/__init__.py`
- `backend/utils/__init__.py`

## 4. `secrets/`

The `secrets/` folder stores local runtime secrets such as Supabase credentials.

Important:

1. Secret values are not documented here.
2. The app reads `secrets/.env` through `3.11.F2 load_env_file()`.
3. `.gitkeep` keeps the folder present in version control.

### 4.1 `secrets/.gitkeep`

Placeholder file so the empty `secrets/` folder can exist in the repository.

## 5. `supabase/`

The `supabase/` folder contains database sql schema.

### 5.1 `supabase/schema.sql`

Provides a consolidated database schema snapshot for the current app state.

What it is for:

1. Creates the full Supabase/PostgreSQL schema in one file.
2. Includes the final table definitions.
3. Defines models, datasets, dataset paths, model artifacts, production sources, monitoring sessions, images, predictions, baseline profiles, drift results, feedback, alerts, and ground-truth labels.
4. Includes indexes, foreign keys, check constraints, unique constraints, and update triggers.

## 6. `tests/`

The `tests/` folder contains pytest coverage for core backend behavior and selected UI helper logic.

### 6.1 `tests/test_core.py`

Tests validation, registration path checks, baseline persistence, session duplicate handling, image overlay math, and drift metrics.

#### 6.1.F1 `make_image(path)`

Creates a small test image.

Calls: none.

#### 6.1.F2 `test_parse_detection_label_file_validates_rows(tmp_path)`

Tests that YOLO detection label parsing accepts valid rows.

Calls: `3.6.F6 parse_detection_label_file()`.

#### 6.1.F3 `test_detection_dataset_images_labels_layout(tmp_path)`

Tests detection dataset validation with image and label folders.

Calls: `6.1.F1 make_image()`, `3.6.F2 validate_detection_dataset()`.

#### 6.1.F4 `test_classification_dataset_labeled_and_unlabeled_test(tmp_path)`

Tests classification dataset validation with labeled train/val folders and an unlabeled test folder.

Calls: `6.1.F1 make_image()`, `3.6.F8 validate_classification_dataset()`.

#### 6.1.F5 `test_drift_fallbacks_work()`

Tests drift helper functions for feature vector and distance fallback behavior.

Calls: `3.5.F9 rbf_mmd()`.

#### 6.1.F6 `test_resolve_dataset_root_accepts_existing_directory(tmp_path)`

Tests that a real directory is accepted as a dataset root.

Calls: `3.10.F2 resolve_dataset_root()`.

#### 6.1.F7 `test_resolve_dataset_root_rejects_file(tmp_path)`

Tests that a file path is rejected as a dataset root.

Calls: `3.10.F2 resolve_dataset_root()`.

#### 6.1.F8 `test_resolve_model_artifact_accepts_pt_file(tmp_path)`

Tests that a `.pt` file is accepted as a model artifact.

Calls: `3.10.F3 resolve_model_artifact()`.

#### 6.1.F9 `test_resolve_model_artifact_rejects_non_pt_file(tmp_path)`

Tests that a non-`.pt` file is rejected as a model artifact.

Calls: `3.10.F3 resolve_model_artifact()`.

#### 6.1.F10 `test_save_baseline_profiles_updates_existing_and_inserts_missing()`

Tests that baseline persistence updates existing profile sections and inserts missing sections.

Calls: `3.3.F1 save_baseline_profiles()`.

#### 6.1.F11 `test_latest_where_orders_and_limits_query()`

Tests that the repository helper orders and limits latest-row queries correctly.

Calls: `3.4.C1 SupabaseRepository`.

#### 6.1.F12 `test_baseline_profile_saves_prediction_reference(monkeypatch, tmp_path)`

Tests that baseline building saves prediction reference information.

Calls: `6.1.F1 make_image()`, `3.2.F1 build_baseline_profile()`.

#### 6.1.F13 `test_session_duplicate_images_are_scoped_to_session(monkeypatch, tmp_path)`

Tests that duplicate image hashes are scoped to each monitoring session.

Calls: `6.1.F1 make_image()`, `3.15.F1 process_source_once()`.

#### 6.1.F14 `test_yolo_box_to_pixel_rect_clips_normalized_detection()`

Tests that normalized YOLO boxes are clipped to image boundaries.

Calls: `2.5.F2 yolo_box_to_pixel_rect()`.

#### 6.1.F15 `test_industrial_prediction_and_concept_metrics()`

Tests Chi-Square prediction drift and human review summary drift helpers.

Calls: `3.5.F13 chi_square_prediction_drift()`, `3.5.F14 review_summary()`.

#### 6.1.F16 `test_prediction_drift_calculates_without_two_feature_rows()`

Tests that prediction drift can still calculate when there are not enough feature rows for total data drift.

Calls: `3.15.F5 calculate_drift_for_window()`.

## Cross-Reference Notes

- UI code in `2 app/` usually calls repository methods from `3.4.C1 SupabaseRepository` and business logic from `3 backend/`.
- Dataset validation starts in `3.6.F1 validate_dataset()` and then routes to detection or classification validation.
- Model artifact metadata starts in `3.8.3.F1 load_yolo_metadata()`.
- YOLO predictions start in `3.8.2.F1 run_yolo_inference()`.
- Baseline creation starts in `3.2.F1 build_baseline_profile()`.
- Live image processing starts in `3.15.F1 process_source_once()`.
- Drift calculation starts in `3.15.F5 calculate_drift_for_window()`.
- Alert generation starts in `3.16.F1 alerts_from_drift()`.
