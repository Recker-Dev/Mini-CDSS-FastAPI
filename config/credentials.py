from google.oauth2 import service_account
import base64
import json
import os

encoded_creds = os.getenv("GOOGLE_CREDENTIALS_BASE64")
# Decode and load the JSON credentials
try:
    decoded_creds = base64.b64decode(encoded_creds).decode("utf-8")
    service_account_info = json.loads(decoded_creds)
    creds = service_account.Credentials.from_service_account_info(service_account_info)
except Exception as e:
    raise ValueError("Failed to decode credentials") from e