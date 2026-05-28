# Secrets

Copy `.env.example` to `.env` and fill in your local credentials.

Never commit `.env` or any real API keys.

Required variables:

- `SUPABASE_URL`
- `SUPABASE_ANON_KEY`

Optional server-side variable:

- `SUPABASE_SERVICE_ROLE_KEY`

The service role key must only be used in trusted backend code. Do not expose it in the Streamlit frontend or browser-accessible code.
