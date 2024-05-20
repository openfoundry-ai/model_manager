import os
import urllib.parse as urlparse
from dotenv import load_dotenv
from supabase import create_client, Client


load_dotenv()

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
if url and key:
    supabase_id = urlparse.urlparse(url).hostname.split('.')[0]
    supabase_client: Client = create_client(url, key)
else:
    supabase_id = None
    supabase_client = None
