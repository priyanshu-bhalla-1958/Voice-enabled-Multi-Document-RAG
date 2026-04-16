from cryptography.fernet import Fernet
import os
from constants import KEY_PATH


def generate_key():
    os.makedirs("storage/keys", exist_ok=True)

    if not os.path.exists(KEY_PATH):
        key = Fernet.generate_key()
        with open(KEY_PATH, "wb") as f:
            f.write(key)

def load_key():
    return open(KEY_PATH, "rb").read()