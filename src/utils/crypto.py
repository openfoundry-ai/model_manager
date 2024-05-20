import os
from cryptography.fernet import Fernet

ENCRYPTION_KEY = os.environ["DATA_ENCRYPTION_KEY"]
key = Fernet(ENCRYPTION_KEY)


def encrypt(value: str):
    return str(key.encrypt(value.encode()), 'utf-8')


def decrypt(value: str):
    return key.decrypt(value)
