from supabase import create_client
from core.config import settings, logger
from typing import Optional, Dict
import asyncio
from functools import partial

# Cache clients by type
_supabase_clients: Dict[str, any] = {}
_init_lock = asyncio.Lock()

async def get_supabase_client(use_service_key=False):
    """
    Initializes and returns the Supabase client (thread-safe).
    Args:
        use_service_key: If True, returns a client using the service role key to bypass RLS
    """
    global _supabase_clients
    
    # Determine client type
    client_type = "service" if use_service_key else "anon"
    
    # Check if we already have this client type initialized
    if client_type not in _supabase_clients:
        async with _init_lock:
            # Double check after acquiring lock
            if client_type not in _supabase_clients:
                url = settings.SUPABASE_URL
                # Choose the appropriate key based on client type
                key = settings.SUPABASE_SERVICE_KEY if use_service_key else settings.SUPABASE_KEY
                
                if url and key:
                    key_type_str = 'service role' if use_service_key else 'anon'
                    logger.info(f"Initializing Supabase client with {key_type_str} key...")
                    try:
                        # Run create_client in a thread pool since it's synchronous
                        loop = asyncio.get_running_loop()
                        client_instance = await loop.run_in_executor(
                            None,
                            partial(create_client, url, key)
                        )
                        
                        # Cache the client
                        _supabase_clients[client_type] = client_instance
                        logger.info(f"Supabase client with {key_type_str} key initialized successfully.")
                    except Exception as e:
                        logger.error(f"Failed to initialize Supabase client with {key_type_str} key: {e}", exc_info=True)
                        raise RuntimeError(f"Failed to initialize Supabase client: {e}")
                else:
                    missing_key = "Service Role Key" if use_service_key else "Anon Key"
                    logger.error(f"Supabase URL or {missing_key} not configured. Cannot create client.")
                    raise ValueError(f"Supabase URL or {missing_key} not configured")
    
    # Return the cached client
    return _supabase_clients[client_type]

# Define Table names here for consistency
PAPERS_TABLE = "papers"
CHUNKS_TABLE = "chunks"