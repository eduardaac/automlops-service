# src/utils/converter.py
import json
import hashlib

def json_2_sha256_key(json_data):
    """Generate SHA256 key from JSON dictionary."""
    json_string = json.dumps(json_data, sort_keys=True)
    sha256_key = hashlib.sha256(json_string.encode("utf-8")).hexdigest()
    return sha256_key