import os
from cryptography.fernet import Fernet
from src.supabase import supabase_client
from src.schemas.secret import Secret

ENCRYPTION_KEY = os.environ["DATA_ENCRYPTION_KEY"]


def set_secrets(user_id: str, secrets: list[Secret]):
    # TODO: dedupe, validate keys
    for secret in secrets:
        key = Fernet(ENCRYPTION_KEY)
        secret_key = secret.key
        secret_value = str(key.encrypt(secret.value.encode()), 'utf-8')
        res = supabase_client.table('access_keys').insert({
            "user_id": user_id,
            "key": secret_key,
            "value": secret_value
        }).execute()
    return


def get_secrets(user_id: str) -> list[Secret]:
    # TODO: optional keys to filter for
    key = Fernet(ENCRYPTION_KEY)

    # RLS for user_id but just in case
    db_secrets = supabase_client.table(
        'access_keys').select("*").eq("user_id", user_id).execute().data
    for secret in db_secrets:
        secrets = [Secret(key=secret["key"], value=key.decrypt(secret["value"]))
                   for secret in db_secrets]
    return secrets
