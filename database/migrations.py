from __future__ import annotations


SCHEMA_VERSION = 1


MIGRATIONS: tuple[str, ...] = (
    """
    CREATE TABLE models (
      id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT, tags TEXT NOT NULL DEFAULT '[]',
      selected_task_type TEXT NOT NULL CHECK(selected_task_type IN ('object_detection','classification')),
      detected_task_type TEXT, task_validation_status TEXT,
      registration_status TEXT NOT NULL DEFAULT 'draft', baseline_status TEXT NOT NULL DEFAULT 'not_started',
      retention_policy TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL, updated_at TEXT NOT NULL
    );
    CREATE TABLE datasets (
      id TEXT PRIMARY KEY, model_id TEXT NOT NULL REFERENCES models(id) ON DELETE CASCADE,
      storage_backend TEXT NOT NULL DEFAULT 'local', source_uri TEXT, storage_path TEXT,
      dataset_root_path TEXT, dataset_type TEXT, dataset_layout TEXT, selected_yaml_path TEXT,
      yaml_filename TEXT, yaml_candidates TEXT NOT NULL DEFAULT '[]', yaml_content TEXT,
      class_source TEXT, num_classes INTEGER, class_names TEXT NOT NULL DEFAULT '[]',
      num_train_images INTEGER NOT NULL DEFAULT 0, num_val_images INTEGER NOT NULL DEFAULT 0,
      num_test_images INTEGER NOT NULL DEFAULT 0, has_test_split INTEGER NOT NULL DEFAULT 0,
      test_has_labels INTEGER NOT NULL DEFAULT 0, supported_image_count INTEGER NOT NULL DEFAULT 0,
      unsupported_file_count INTEGER NOT NULL DEFAULT 0, validation_status TEXT,
      validation_errors TEXT NOT NULL DEFAULT '[]', created_at TEXT NOT NULL, updated_at TEXT NOT NULL
    );
    CREATE TABLE dataset_paths (
      id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
      split TEXT NOT NULL CHECK(split IN ('train','val','test')), images_path TEXT, labels_path TEXT,
      has_images INTEGER NOT NULL DEFAULT 0, has_labels INTEGER NOT NULL DEFAULT 0,
      image_count INTEGER NOT NULL DEFAULT 0, label_count INTEGER NOT NULL DEFAULT 0,
      created_at TEXT NOT NULL, UNIQUE(dataset_id, split)
    );
    CREATE TABLE model_artifacts (
      id TEXT PRIMARY KEY, model_id TEXT NOT NULL REFERENCES models(id) ON DELETE CASCADE,
      storage_backend TEXT NOT NULL DEFAULT 'local', source_uri TEXT, storage_path TEXT, local_path TEXT,
      artifact_name TEXT NOT NULL DEFAULT 'best.pt', artifact_type TEXT NOT NULL DEFAULT 'yolo_pt',
      model_task TEXT, class_names TEXT NOT NULL DEFAULT '[]', num_classes INTEGER, input_size INTEGER,
      is_compatible INTEGER NOT NULL DEFAULT 0, compatibility_status TEXT,
      compatibility_details TEXT NOT NULL DEFAULT '{}', raw_metadata TEXT NOT NULL DEFAULT '{}',
      artifact_sha256 TEXT, artifact_size_bytes INTEGER, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
    );
    CREATE TABLE production_sources (
      id TEXT PRIMARY KEY, model_id TEXT NOT NULL REFERENCES models(id) ON DELETE CASCADE,
      source_type TEXT NOT NULL CHECK(source_type IN ('test_folder','manual_upload','watched_folder')),
      source_uri TEXT, label_uri TEXT, mode TEXT NOT NULL, polling_interval_seconds INTEGER NOT NULL DEFAULT 2,
      is_active INTEGER NOT NULL DEFAULT 1, config TEXT NOT NULL DEFAULT '{}',
      created_at TEXT NOT NULL, updated_at TEXT NOT NULL
    );
    CREATE TABLE monitoring_sessions (
      id TEXT PRIMARY KEY, model_id TEXT NOT NULL REFERENCES models(id) ON DELETE CASCADE,
      dataset_id TEXT REFERENCES datasets(id) ON DELETE SET NULL,
      artifact_id TEXT REFERENCES model_artifacts(id) ON DELETE SET NULL,
      source_id TEXT REFERENCES production_sources(id) ON DELETE SET NULL,
      source_type TEXT, source_uri TEXT, polling_interval_seconds INTEGER, name TEXT,
      status TEXT NOT NULL DEFAULT 'created', worker_status TEXT, worker_owner_id TEXT, worker_pid INTEGER,
      worker_heartbeat_at TEXT, worker_lease_expires_at TEXT, last_processed_at TEXT,
      processing_started_at TEXT, processing_attempt_count INTEGER NOT NULL DEFAULT 0,
      last_worker_error TEXT, completed_reason TEXT, diagnostic_status TEXT,
      diagnostic_triggered_at TEXT, diagnostic_window_start TEXT, diagnostic_window_end TEXT,
      diagnostic_reason TEXT, diagnostic_error TEXT, threshold_config TEXT NOT NULL DEFAULT '{}',
      summary TEXT NOT NULL DEFAULT '{}', started_at TEXT NOT NULL, ended_at TEXT, archived_at TEXT,
      last_processed_image_id TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
    );
    CREATE TABLE images (
      id TEXT PRIMARY KEY, model_id TEXT NOT NULL REFERENCES models(id) ON DELETE CASCADE,
      dataset_id TEXT REFERENCES datasets(id) ON DELETE SET NULL,
      session_id TEXT REFERENCES monitoring_sessions(id) ON DELETE SET NULL,
      source TEXT, split TEXT, storage_path TEXT, local_path TEXT, filename TEXT NOT NULL,
      content_hash TEXT, image_format TEXT, color_mode TEXT, num_channels INTEGER, bit_depth INTEGER,
      width INTEGER, height INTEGER, aspect_ratio REAL, file_size_bytes INTEGER,
      brightness_mean REAL, brightness_std REAL, contrast REAL, sharpness REAL,
      saturation_mean REAL, saturation_std REAL, edge_density REAL,
      feature_payload TEXT NOT NULL DEFAULT '{}', inference_status TEXT NOT NULL DEFAULT 'complete',
      prediction_count INTEGER NOT NULL DEFAULT 0, inference_error TEXT,
      inference_completed_at TEXT, created_at TEXT NOT NULL
    );
    CREATE TABLE ground_truth_labels (
      id TEXT PRIMARY KEY, image_id TEXT NOT NULL REFERENCES images(id) ON DELETE CASCADE,
      task_type TEXT NOT NULL, class_id INTEGER, class_name TEXT, x_center REAL, y_center REAL,
      width REAL, height REAL, source TEXT NOT NULL, created_at TEXT NOT NULL
    );
    CREATE TABLE predictions (
      id TEXT PRIMARY KEY, image_id TEXT NOT NULL REFERENCES images(id) ON DELETE CASCADE,
      artifact_id TEXT REFERENCES model_artifacts(id) ON DELETE SET NULL, task_type TEXT NOT NULL,
      predicted_class_id INTEGER, predicted_class_name TEXT, confidence REAL, top_k TEXT,
      x_center REAL, y_center REAL, width REAL, height REAL, inference_time_ms REAL,
      raw_prediction TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL
    );
    CREATE TABLE baseline_profiles (
      id TEXT PRIMARY KEY, model_id TEXT NOT NULL REFERENCES models(id) ON DELETE CASCADE,
      artifact_id TEXT REFERENCES model_artifacts(id) ON DELETE SET NULL,
      dataset_id TEXT REFERENCES datasets(id) ON DELETE SET NULL,
      profile_type TEXT NOT NULL, metrics TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL,
      UNIQUE(model_id, dataset_id, artifact_id, profile_type)
    );
    CREATE TABLE drift_results (
      id TEXT PRIMARY KEY, model_id TEXT NOT NULL REFERENCES models(id) ON DELETE CASCADE,
      session_id TEXT REFERENCES monitoring_sessions(id) ON DELETE CASCADE,
      window_start TEXT, window_end TEXT, num_images INTEGER, drift_type TEXT NOT NULL,
      metric_name TEXT NOT NULL, metric_value REAL, threshold REAL, status TEXT NOT NULL,
      details TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL
    );
    CREATE TABLE feedback (
      id TEXT PRIMARY KEY, image_id TEXT NOT NULL REFERENCES images(id) ON DELETE CASCADE,
      prediction_id TEXT REFERENCES predictions(id) ON DELETE SET NULL, feedback_type TEXT NOT NULL,
      corrected_payload TEXT NOT NULL DEFAULT '{}', comment TEXT, created_at TEXT NOT NULL
    );
    CREATE TABLE alerts (
      id TEXT PRIMARY KEY, model_id TEXT NOT NULL REFERENCES models(id) ON DELETE CASCADE,
      session_id TEXT REFERENCES monitoring_sessions(id) ON DELETE CASCADE,
      severity TEXT NOT NULL, title TEXT NOT NULL, message TEXT,
      drift_result_id TEXT REFERENCES drift_results(id) ON DELETE SET NULL,
      is_resolved INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL,
      last_seen_at TEXT, resolved_at TEXT
    );
    CREATE INDEX idx_models_status ON models(registration_status, baseline_status);
    CREATE INDEX idx_datasets_model ON datasets(model_id, created_at DESC);
    CREATE INDEX idx_dataset_paths_dataset ON dataset_paths(dataset_id, split);
    CREATE INDEX idx_artifacts_model ON model_artifacts(model_id, created_at DESC);
    CREATE INDEX idx_sources_model_active ON production_sources(model_id, is_active, created_at DESC);
    CREATE INDEX idx_sessions_status_created ON monitoring_sessions(status, created_at DESC);
    CREATE INDEX idx_sessions_model_status ON monitoring_sessions(model_id, status, created_at DESC);
    CREATE INDEX idx_sessions_worker ON monitoring_sessions(worker_owner_id, status);
    CREATE INDEX idx_images_session_created ON images(session_id, created_at DESC);
    CREATE INDEX idx_images_session_status ON images(session_id, inference_status, created_at DESC);
    CREATE UNIQUE INDEX idx_images_unique_content
      ON images(model_id, session_id, content_hash) WHERE session_id IS NOT NULL AND content_hash IS NOT NULL;
    CREATE INDEX idx_predictions_image_artifact ON predictions(image_id, artifact_id);
    CREATE INDEX idx_baselines_lookup ON baseline_profiles(model_id, dataset_id, artifact_id, profile_type);
    CREATE INDEX idx_drift_history ON drift_results(session_id, metric_name, created_at DESC);
    CREATE UNIQUE INDEX idx_drift_unique_window
      ON drift_results(session_id, metric_name, window_end) WHERE window_end IS NOT NULL;
    CREATE INDEX idx_feedback_image ON feedback(image_id, created_at DESC);
    CREATE INDEX idx_alerts_session_resolved ON alerts(session_id, is_resolved, created_at DESC);
    CREATE UNIQUE INDEX idx_alerts_unique_open
      ON alerts(session_id, title, severity) WHERE is_resolved = 0;
    """,
)
