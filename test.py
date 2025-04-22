import os
from supabase import create_client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Use service_role key instead of anon key
url = os.getenv("SUPABASE_URL")
service_key = os.getenv("SUPABASE_SERVICE_KEY")  # Service role key from dashboard

# Initialize Supabase client with service role key
supabase = create_client(url, service_key)

try:
  
    
    # Test by uploading a file
    with open("test.pdf", "rb") as f:
        response = (
            supabase.storage
            .from_("irn-documents")
            .upload(
                path="test.pdf",
                file=f,
                file_options={"content-type": "application/pdf"}
            )
        )
    print("File uploaded successfully!")
    
    # Get the public URL
    public_url = supabase.storage.from_("irn-documents").get_public_url("test.pdf")
    print(f"Public URL: {public_url}")
    
except Exception as e:
    print(f"Error: {e}")
