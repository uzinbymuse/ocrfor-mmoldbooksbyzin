#!/usr/bin/env python3
"""Auto-updater for mya_dhamma traineddata from GitHub (uzinbymuse/ocrfor-mmoldbooksbyzin)."""
import os
import json
import urllib.request

REPO_RAW_BASE = "https://raw.githubusercontent.com/uzinbymuse/ocrfor-mmoldbooksbyzin/main"
MANIFEST_URL = f"{REPO_RAW_BASE}/version.json"
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESSDATA_DIR = os.path.join(ROOT_DIR, "tessdata")
LOCAL_VERSION_FILE = os.path.join(TESSDATA_DIR, "local_version.json")


def get_local_info():
    if os.path.exists(LOCAL_VERSION_FILE):
        with open(LOCAL_VERSION_FILE, encoding="utf-8") as f:
            return json.load(f)
    # Fallback: detect newest mya_dhamma*.traineddata in tessdata/
    os.makedirs(TESSDATA_DIR, exist_ok=True)
    files = sorted([f for f in os.listdir(TESSDATA_DIR) if f.endswith(".traineddata")])
    if files:
        latest = files[-1]
        return {"version": "local", "model_name": os.path.splitext(latest)[0], "filename": latest}
    return {"version": "0.0.0", "model_name": "mya_dhamma_v2", "filename": "mya_dhamma_v2.traineddata"}


def get_active_model_name():
    info = get_local_info()
    target = os.path.join(TESSDATA_DIR, info.get("filename", "mya_dhamma_v2.traineddata"))
    if os.path.exists(target):
        return info["model_name"]
    return "mya"


def check_and_update(log_cb=print):
    """Checks GitHub version.json and downloads newer traineddata if available."""
    os.makedirs(TESSDATA_DIR, exist_ok=True)
    local = get_local_info()
    log_cb(f"Checking GitHub for model updates (current: {local.get('model_name')} v{local.get('version')})...")
    try:
        req = urllib.request.Request(MANIFEST_URL, headers={"User-Agent": "ocrfor-mmoldbooksbyzin-updater"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            remote = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        log_cb(f"Update check failed (offline or repo unreachable): {e}")
        return False, local

    target_path = os.path.join(TESSDATA_DIR, remote["filename"])
    needs_dl = (remote["version"] != local.get("version")) or (not os.path.exists(target_path))
    if not needs_dl:
        log_cb(f"Already up to date: {remote['model_name']} (v{remote['version']})")
        return False, remote

    log_cb(f"Downloading new model: {remote['filename']} (v{remote['version']})...")
    try:
        tmp_path = target_path + ".part"
        req = urllib.request.Request(remote["download_url"], headers={"User-Agent": "ocrfor-mmoldbooksbyzin-updater"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(tmp_path, "wb") as out:
            out.write(resp.read())
        os.replace(tmp_path, target_path)
        with open(LOCAL_VERSION_FILE, "w", encoding="utf-8") as f:
            json.dump(remote, f, ensure_ascii=False, indent=2)
        log_cb(f"Updated successfully -> {remote['model_name']} (v{remote['version']})")
        return True, remote
    except Exception as e:
        log_cb(f"Model download failed: {e}")
        return False, local


if __name__ == "__main__":
    check_and_update()
