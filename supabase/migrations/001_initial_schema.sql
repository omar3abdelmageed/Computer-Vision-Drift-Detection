create extension if not exists "pgcrypto";

create table if not exists models (
  id uuid primary key default gen_random_uuid(),
  name text not null,
  description text,
  tags jsonb default '[]'::jsonb,
  selected_task_type text not null check (selected_task_type in ('object_detection', 'classification')),
  detected_task_type text check (detected_task_type in ('object_detection', 'classification', 'ambiguous', 'invalid')),
  task_validation_status text check (task_validation_status in ('valid', 'warning', 'invalid')),
  registration_status text not null default 'draft' check (registration_status in ('draft', 'dataset_uploaded', 'dataset_validated', 'artifact_uploaded', 'compatible', 'baseline_running', 'baseline_completed', 'failed')),
  baseline_status text not null default 'not_started' check (baseline_status in ('not_started', 'running', 'completed', 'failed')),
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

create index if not exists idx_models_selected_task_type on models(selected_task_type);
create index if not exists idx_models_registration_status on models(registration_status);
create index if not exists idx_models_baseline_status on models(baseline_status);

create table if not exists datasets (
  id uuid primary key default gen_random_uuid(),
  model_id uuid not null references models(id) on delete cascade,
  storage_path text,
  dataset_root_path text,
  dataset_type text check (dataset_type in ('yolo_detection', 'yolo_classification')),
  dataset_layout text check (dataset_layout in ('detection_images_labels_root', 'detection_split_first', 'classification_split_first', 'classification_images_root', 'classification_unlabeled_test', 'unknown')),
  selected_yaml_path text,
  yaml_filename text,
  yaml_candidates jsonb default '[]'::jsonb,
  yaml_content jsonb,
  class_source text check (class_source in ('yaml_names', 'folder_names', 'model_names', 'unknown')),
  num_classes int,
  class_names jsonb default '[]'::jsonb,
  num_train_images int default 0,
  num_val_images int default 0,
  num_test_images int default 0,
  has_test_split boolean default false,
  test_has_labels boolean default false,
  supported_image_count int default 0,
  unsupported_file_count int default 0,
  validation_status text check (validation_status in ('valid', 'warning', 'invalid')),
  validation_errors jsonb default '[]'::jsonb,
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

create index if not exists idx_datasets_model_id on datasets(model_id);
create index if not exists idx_datasets_validation_status on datasets(validation_status);
create index if not exists idx_datasets_layout on datasets(dataset_layout);

create table if not exists dataset_paths (
  id uuid primary key default gen_random_uuid(),
  dataset_id uuid not null references datasets(id) on delete cascade,
  split text not null check (split in ('train', 'val', 'test')),
  images_path text,
  labels_path text,
  has_images boolean default false,
  has_labels boolean default false,
  image_count int default 0,
  label_count int default 0,
  created_at timestamptz default now(),
  unique(dataset_id, split)
);

create index if not exists idx_dataset_paths_dataset_id on dataset_paths(dataset_id);
create index if not exists idx_dataset_paths_split on dataset_paths(split);

create table if not exists model_artifacts (
  id uuid primary key default gen_random_uuid(),
  model_id uuid not null references models(id) on delete cascade,
  storage_path text not null,
  local_path text,
  artifact_name text not null default 'best.pt',
  artifact_type text not null default 'yolo_pt' check (artifact_type in ('yolo_pt', 'onnx', 'torchscript', 'other')),
  model_task text check (model_task in ('detect', 'classify', 'segment', 'pose', 'obb', 'unknown')),
  class_names jsonb default '[]'::jsonb,
  num_classes int,
  input_size int,
  is_compatible boolean default false,
  compatibility_status text check (compatibility_status in ('valid', 'warning', 'invalid')),
  compatibility_details jsonb default '{}'::jsonb,
  raw_metadata jsonb default '{}'::jsonb,
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

create index if not exists idx_model_artifacts_model_id on model_artifacts(model_id);
create index if not exists idx_model_artifacts_compatibility_status on model_artifacts(compatibility_status);

create table if not exists images (
  id uuid primary key default gen_random_uuid(),
  model_id uuid not null references models(id) on delete cascade,
  dataset_id uuid references datasets(id) on delete set null,
  session_id uuid,
  source text check (source in ('train', 'val', 'test', 'production', 'manual_upload', 'watched_folder', 'camera', 'supabase_bucket')),
  split text check (split in ('train', 'val', 'test', 'production')),
  storage_path text,
  local_path text,
  filename text not null,
  content_hash text,
  image_format text,
  color_mode text,
  num_channels int,
  bit_depth int,
  width int,
  height int,
  aspect_ratio float,
  file_size_bytes bigint,
  brightness_mean float,
  brightness_std float,
  contrast float,
  sharpness float,
  saturation_mean float,
  saturation_std float,
  edge_density float,
  feature_payload jsonb default '{}'::jsonb,
  created_at timestamptz default now(),
  unique(model_id, content_hash)
);

create index if not exists idx_images_model_id on images(model_id);
create index if not exists idx_images_dataset_id on images(dataset_id);
create index if not exists idx_images_session_id on images(session_id);
create index if not exists idx_images_source on images(source);
create index if not exists idx_images_split on images(split);
create index if not exists idx_images_content_hash on images(content_hash);

create table if not exists ground_truth_labels (
  id uuid primary key default gen_random_uuid(),
  image_id uuid not null references images(id) on delete cascade,
  task_type text not null check (task_type in ('object_detection', 'classification')),
  class_id int,
  class_name text,
  x_center float,
  y_center float,
  width float,
  height float,
  source text not null check (source in ('dataset', 'human_feedback', 'delayed_ground_truth')),
  created_at timestamptz default now()
);

create index if not exists idx_ground_truth_labels_image_id on ground_truth_labels(image_id);
create index if not exists idx_ground_truth_labels_task_type on ground_truth_labels(task_type);
create index if not exists idx_ground_truth_labels_class_id on ground_truth_labels(class_id);
create index if not exists idx_ground_truth_labels_source on ground_truth_labels(source);

create table if not exists predictions (
  id uuid primary key default gen_random_uuid(),
  image_id uuid not null references images(id) on delete cascade,
  artifact_id uuid references model_artifacts(id) on delete set null,
  task_type text not null check (task_type in ('object_detection', 'classification')),
  predicted_class_id int,
  predicted_class_name text,
  confidence float,
  top_k jsonb,
  x_center float,
  y_center float,
  width float,
  height float,
  inference_time_ms float,
  raw_prediction jsonb default '{}'::jsonb,
  created_at timestamptz default now()
);

create index if not exists idx_predictions_image_id on predictions(image_id);
create index if not exists idx_predictions_artifact_id on predictions(artifact_id);
create index if not exists idx_predictions_task_type on predictions(task_type);
create index if not exists idx_predictions_predicted_class_id on predictions(predicted_class_id);
create index if not exists idx_predictions_confidence on predictions(confidence);

create table if not exists baseline_profiles (
  id uuid primary key default gen_random_uuid(),
  model_id uuid not null references models(id) on delete cascade,
  artifact_id uuid references model_artifacts(id) on delete set null,
  dataset_id uuid references datasets(id) on delete set null,
  profile_type text not null check (profile_type in ('dataset_profile', 'image_stats', 'embeddings', 'predictions', 'performance', 'object_distribution', 'class_distribution')),
  metrics jsonb not null default '{}'::jsonb,
  created_at timestamptz default now()
);

create index if not exists idx_baseline_profiles_model_id on baseline_profiles(model_id);
create index if not exists idx_baseline_profiles_artifact_id on baseline_profiles(artifact_id);
create index if not exists idx_baseline_profiles_dataset_id on baseline_profiles(dataset_id);
create index if not exists idx_baseline_profiles_profile_type on baseline_profiles(profile_type);

create table if not exists production_sources (
  id uuid primary key default gen_random_uuid(),
  model_id uuid not null references models(id) on delete cascade,
  source_type text not null check (source_type in ('test_folder', 'manual_upload', 'watched_folder', 'supabase_bucket', 'rtsp', 'usb_camera')),
  source_uri text,
  label_uri text,
  mode text not null check (mode in ('initial_evaluation', 'live_monitoring', 'manual_batch')),
  polling_interval_seconds int default 30,
  is_active boolean default true,
  config jsonb default '{}'::jsonb,
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

create index if not exists idx_production_sources_model_id on production_sources(model_id);
create index if not exists idx_production_sources_source_type on production_sources(source_type);
create index if not exists idx_production_sources_is_active on production_sources(is_active);

create table if not exists monitoring_sessions (
  id uuid primary key default gen_random_uuid(),
  model_id uuid not null references models(id) on delete cascade,
  artifact_id uuid references model_artifacts(id) on delete set null,
  source_id uuid references production_sources(id) on delete set null,
  source_type text check (source_type in ('test_folder', 'manual_upload', 'watched_folder', 'supabase_bucket', 'rtsp', 'usb_camera')),
  status text not null default 'created' check (status in ('created', 'waiting_for_live_data', 'running_test_evaluation', 'running_live_monitoring', 'paused', 'completed', 'failed')),
  started_at timestamptz default now(),
  ended_at timestamptz,
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

create index if not exists idx_monitoring_sessions_model_id on monitoring_sessions(model_id);
create index if not exists idx_monitoring_sessions_artifact_id on monitoring_sessions(artifact_id);
create index if not exists idx_monitoring_sessions_source_id on monitoring_sessions(source_id);
create index if not exists idx_monitoring_sessions_status on monitoring_sessions(status);

alter table images drop constraint if exists fk_images_session_id;
alter table images add constraint fk_images_session_id foreign key (session_id) references monitoring_sessions(id) on delete set null;

create table if not exists drift_results (
  id uuid primary key default gen_random_uuid(),
  model_id uuid not null references models(id) on delete cascade,
  session_id uuid references monitoring_sessions(id) on delete cascade,
  window_start timestamptz,
  window_end timestamptz,
  num_images int,
  drift_type text not null check (drift_type in ('data', 'prediction', 'concept')),
  metric_name text not null,
  metric_value float,
  threshold float,
  status text not null check (status in ('ok', 'warning', 'critical')),
  details jsonb default '{}'::jsonb,
  created_at timestamptz default now()
);

create index if not exists idx_drift_results_model_id on drift_results(model_id);
create index if not exists idx_drift_results_session_id on drift_results(session_id);
create index if not exists idx_drift_results_drift_type on drift_results(drift_type);
create index if not exists idx_drift_results_metric_name on drift_results(metric_name);
create index if not exists idx_drift_results_status on drift_results(status);

create table if not exists feedback (
  id uuid primary key default gen_random_uuid(),
  image_id uuid not null references images(id) on delete cascade,
  prediction_id uuid references predictions(id) on delete set null,
  reviewer_id uuid,
  feedback_type text not null check (feedback_type in ('approve', 'reject', 'corrected_label', 'corrected_box', 'missed_object', 'false_positive', 'wrong_class')),
  corrected_payload jsonb default '{}'::jsonb,
  comment text,
  created_at timestamptz default now()
);

create index if not exists idx_feedback_image_id on feedback(image_id);
create index if not exists idx_feedback_prediction_id on feedback(prediction_id);
create index if not exists idx_feedback_feedback_type on feedback(feedback_type);

create table if not exists alerts (
  id uuid primary key default gen_random_uuid(),
  model_id uuid not null references models(id) on delete cascade,
  session_id uuid references monitoring_sessions(id) on delete cascade,
  severity text not null check (severity in ('info', 'warning', 'critical')),
  title text not null,
  message text,
  drift_result_id uuid references drift_results(id) on delete set null,
  is_resolved boolean default false,
  created_at timestamptz default now(),
  resolved_at timestamptz
);

create index if not exists idx_alerts_model_id on alerts(model_id);
create index if not exists idx_alerts_session_id on alerts(session_id);
create index if not exists idx_alerts_severity on alerts(severity);
create index if not exists idx_alerts_is_resolved on alerts(is_resolved);

create or replace function set_updated_at()
returns trigger as $$
begin
  new.updated_at = now();
  return new;
end;
$$ language plpgsql;

drop trigger if exists trg_models_updated_at on models;
create trigger trg_models_updated_at before update on models for each row execute function set_updated_at();
drop trigger if exists trg_datasets_updated_at on datasets;
create trigger trg_datasets_updated_at before update on datasets for each row execute function set_updated_at();
drop trigger if exists trg_model_artifacts_updated_at on model_artifacts;
create trigger trg_model_artifacts_updated_at before update on model_artifacts for each row execute function set_updated_at();
drop trigger if exists trg_production_sources_updated_at on production_sources;
create trigger trg_production_sources_updated_at before update on production_sources for each row execute function set_updated_at();
drop trigger if exists trg_monitoring_sessions_updated_at on monitoring_sessions;
create trigger trg_monitoring_sessions_updated_at before update on monitoring_sessions for each row execute function set_updated_at();
