import sys, os
sys.path.insert(0, '.')
os.chdir(os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.')

from dotenv import load_dotenv
load_dotenv()

from azure_clients.ai_search_client import ai_search

results = ai_search.search("revenue", query_vector=[0.0]*1536, top_k=1)
r = results[0] if results else {}
print("has embedding:", "embedding" in r)
print("keys:", [k for k in r.keys() if not k.startswith('@')])