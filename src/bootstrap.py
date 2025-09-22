# src/bootstrap.py
# -*- coding: utf-8 -*-
import os, hashlib, subprocess, sys
from typing import Tuple

def _sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def ensure_requirements_installed(requirements_path: str = "requirements.txt",
                                  lock_path: str = ".requirements.sha256") -> Tuple[bool, str]:
    try:
        if not os.path.exists(requirements_path):
            return False, f"requirements file not found: {requirements_path}"
        new_hash = _sha256_of_file(requirements_path)
        old_hash = None
        if os.path.exists(lock_path):
            try:
                with open(lock_path, "r", encoding="utf-8") as f:
                    old_hash = f.read().strip()
            except Exception:
                old_hash = None
        if new_hash != old_hash:
            cmd = [sys.executable, "-m", "pip", "install", "-r", requirements_path, "--upgrade"]
            env = os.environ.copy()
            env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
            if proc.returncode != 0:
                return False, f"pip install failed: {proc.stderr[:500]}"
            with open(lock_path, "w", encoding="utf-8") as f:
                f.write(new_hash)
            return True, "installed/updated from requirements.txt"
        return True, "requirements already satisfied"
    except Exception as e:
        return False, f"ensure_requirements_installed error: {e}"
