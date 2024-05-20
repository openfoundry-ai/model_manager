from src.supabase import supabase_client
from src.schemas.secret import Secret
from src.utils.crypto import encrypt, decrypt
from pydantic import SecretStr


def set_secrets(user_id: str, secrets: list[Secret]):
    data = [{
            'user_id': user_id,
            'key': secret.key,
            'value': encrypt(secret.value.get_secret_value())
            } for secret in secrets]

    res = supabase_client.table('access_keys').upsert(data).execute()
    return


def get_secrets(user_id: str) -> list[Secret]:
    # TODO: optional keys to filter for

    # RLS for user_id but just in case
    db_secrets = supabase_client.table(
        'access_keys').select('*').eq('user_id', user_id).execute().data

    secrets = [Secret(key=secret['key'], value=SecretStr(decrypt(secret['value'])))
               for secret in db_secrets]
    return secrets
