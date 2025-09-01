# gen_admin_hash.py
import hashlib, secrets, getpass

def gen(password: str):
    salt = secrets.token_hex(16)                 # 32 hex chars (random)
    pw_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode(),
        salt.encode(),
        200_000
    ).hex()
    return salt, pw_hash

if __name__ == "__main__":
    pwd = getpass.getpass("Admin password (will not echo): ")
    s, h = gen(pwd)
    print("SALT:", s)
    print("HASH:", h)
