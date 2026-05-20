from src.database.client import supabase

def get_models_by_user(user_id: str):
    """Retrieve all models uploaded by a specific user."""
    response = supabase.table("models").select("*").eq("user_id", user_id).execute()
    return response.data

def insert_model(model_data: dict):
    """Insert a new model record."""
    response = supabase.table("models").insert(model_data).execute()
    return response.data

def get_drift_metrics(model_id: str):
    """Retrieve all drift metrics for a specific model over time."""
    response = supabase.table("drift_metrics").select("*").eq("model_id", model_id).order("timestamp", desc=False).execute()
    return response.data

def insert_drift_metrics(metrics_data: dict):
    """Insert a new drift metric record for a model."""
    response = supabase.table("drift_metrics").insert(metrics_data).execute()
    return response.data

def upload_file_to_storage(bucket: str, file_path: str, file_bytes: bytes):
    """Upload a file to Supabase Storage."""
    response = supabase.storage.from_(bucket).upload(file_path, file_bytes)
    return response

def get_file_url(bucket: str, file_path: str):
    """Get the public URL for a file in storage."""
    return supabase.storage.from_(bucket).get_public_url(file_path)
