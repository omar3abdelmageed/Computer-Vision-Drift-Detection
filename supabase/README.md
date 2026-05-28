# Supabase Setup

Apply the SQL files in `migrations/` to your Supabase Postgres database in order.

Large datasets and YOLO `.pt` model artifacts are local-first. Keep them on the filesystem available to the Streamlit process, then register their absolute paths in the app.

Supabase Storage buckets are not required for dataset or model artifact registration. They can still be added later for small assets or reports.

RLS is intentionally not enabled for local development. Add ownership columns and policies before production deployment.
