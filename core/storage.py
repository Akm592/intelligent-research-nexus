# core/storage.py
"""
Core Storage Utilities.

This module is intended for housing functions and classes related to interactions
with file storage systems, such as cloud storage (e.g., Supabase Storage, AWS S3,
Google Cloud Storage) or local file storage abstractions.

Currently, direct file uploads in the UI service use Supabase client library calls.
This module could be developed to centralize such storage operations, provide
consistent error handling, or support different storage backends.
"""
