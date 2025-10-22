# modules/config.py
import os, re

# Override per machine: export PROJECT_DIR="/users/project1/pt01183"
PROJECT_DIR = os.environ.get("PROJECT_DIR", "/mnt/project/pt01183")

def city_to_dir(city: str) -> str:
    s = city.replace("/", "_").replace("\\", "_")
    s = re.sub(r"\s+", " ", s).strip()
    return s
