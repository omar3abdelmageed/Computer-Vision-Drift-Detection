alter table datasets
add column if not exists storage_backend text not null default 'local';

alter table datasets
add column if not exists source_uri text;

do $$
begin
  if not exists (
    select 1 from pg_constraint where conname = 'datasets_storage_backend_check'
  ) then
    alter table datasets
    add constraint datasets_storage_backend_check
    check (storage_backend in ('local', 'supabase'));
  end if;
end $$;

alter table model_artifacts
add column if not exists storage_backend text not null default 'local';

alter table model_artifacts
add column if not exists source_uri text;

alter table model_artifacts
alter column storage_path drop not null;

do $$
begin
  if not exists (
    select 1 from pg_constraint where conname = 'model_artifacts_storage_backend_check'
  ) then
    alter table model_artifacts
    add constraint model_artifacts_storage_backend_check
    check (storage_backend in ('local', 'supabase'));
  end if;
end $$;
