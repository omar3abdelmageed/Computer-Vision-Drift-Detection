-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 1. Create Models Table
CREATE TABLE IF NOT EXISTS public.models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    model_type TEXT NOT NULL, -- e.g., 'Classification', 'Object Detection'
    classes JSONB, -- list of class labels
    thresholds JSONB, -- drift thresholds config
    storage_path TEXT, -- path to model file in Supabase Storage
    dataset_path TEXT, -- path to baseline dataset in Supabase Storage
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enable RLS
ALTER TABLE public.models ENABLE ROW LEVEL SECURITY;

-- Policies for models
CREATE POLICY "Users can view their own models" ON public.models
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own models" ON public.models
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete their own models" ON public.models
    FOR DELETE USING (auth.uid() = user_id);

CREATE POLICY "Users can update their own models" ON public.models
    FOR UPDATE USING (auth.uid() = user_id);

-- 2. Create Drift Metrics Table
CREATE TABLE IF NOT EXISTS public.drift_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID NOT NULL REFERENCES public.models(id) ON DELETE CASCADE,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    data_drift_score NUMERIC,
    prediction_drift_score NUMERIC,
    concept_drift_score NUMERIC,
    details JSONB -- Detailed JSON of test outputs (p-values, distances, etc.)
);

-- Enable RLS
ALTER TABLE public.drift_metrics ENABLE ROW LEVEL SECURITY;

-- Policies for drift metrics
CREATE POLICY "Users can view metrics for their models" ON public.drift_metrics
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM public.models 
            WHERE models.id = drift_metrics.model_id AND models.user_id = auth.uid()
        )
    );

CREATE POLICY "Users can insert metrics for their models" ON public.drift_metrics
    FOR INSERT WITH CHECK (
        EXISTS (
            SELECT 1 FROM public.models 
            WHERE models.id = drift_metrics.model_id AND models.user_id = auth.uid()
        )
    );
