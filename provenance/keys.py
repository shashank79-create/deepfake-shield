"""
provenance/keys.py
Ed25519 signing keys.

The PRIVATE key never leaves your machine and must be kept secret: anyone who
has it can sign images as you. The PUBLIC key is what you publish (website,
bio, GitHub) so others can verify your photos.
"""

import base64
import hashlib
from pathlib import Path
from typing import Optional, Tuple

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey


def key_id(public_key: Ed25519PublicKey) -> str:
    """Short fingerprint people can compare by eye: first 16 hex chars of SHA-256(raw key)."""
    return hashlib.sha256(raw_public(public_key)).hexdigest()[:16]


def raw_public(public_key: Ed25519PublicKey) -> bytes:
    return public_key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


def public_from_b64(b64: str) -> Ed25519PublicKey:
    return Ed25519PublicKey.from_public_bytes(base64.b64decode(b64))


def generate_keypair(out_dir: str, name: str = "signer", password: Optional[str] = None) -> Tuple[Path, Path]:
    """Write <name>.key (private, PEM, optionally password-encrypted) and <name>.pub (public, PEM)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    priv_path, pub_path = out / f"{name}.key", out / f"{name}.pub"
    if priv_path.exists():
        raise FileExistsError(f"{priv_path} already exists - refusing to overwrite a signing key.")
    sk = Ed25519PrivateKey.generate()
    enc = (serialization.BestAvailableEncryption(password.encode()) if password
           else serialization.NoEncryption())
    priv_path.write_bytes(sk.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, enc))
    pub_path.write_bytes(sk.public_key().public_bytes(serialization.Encoding.PEM,
                                                      serialization.PublicFormat.SubjectPublicKeyInfo))
    try:
        priv_path.chmod(0o600)
    except OSError:
        pass
    return priv_path, pub_path


def load_private(path: str, password: Optional[str] = None) -> Ed25519PrivateKey:
    sk = serialization.load_pem_private_key(Path(path).read_bytes(),
                                            password=password.encode() if password else None)
    if not isinstance(sk, Ed25519PrivateKey):
        raise ValueError("Not an Ed25519 private key")
    return sk


def load_public(path: str) -> Ed25519PublicKey:
    pk = serialization.load_pem_public_key(Path(path).read_bytes())
    if not isinstance(pk, Ed25519PublicKey):
        raise ValueError("Not an Ed25519 public key")
    return pk
