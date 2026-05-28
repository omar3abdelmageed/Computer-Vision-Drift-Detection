alter table baseline_profiles
drop constraint if exists baseline_profiles_profile_type_check;

alter table baseline_profiles
add constraint baseline_profiles_profile_type_check
check (
  profile_type in (
    'dataset_profile',
    'image_stats',
    'feature_rows',
    'embeddings',
    'predictions',
    'performance',
    'object_distribution',
    'class_distribution'
  )
);
