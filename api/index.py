"""
Vercel Serverless Entry Point

Exports the FastAPI app for Vercel Python runtime.
"""

from main import app

# Vercel requires the app to be named 'app' or 'handler'
handler = app
