
from comfy_models import MODEL_LIST
import os

print(f"Checking {len(MODEL_LIST)} models...")

for i, model in enumerate(MODEL_LIST):
    print(f"Checking model {i+1}...")
    if not isinstance(model.get("url"), str):
        print(f"ERROR: Model {i+1} url is not a string: {model.get('url')}")
    if not isinstance(model.get("path"), str):
        print(f"ERROR: Model {i+1} path is not a string: {model.get('path')}")
    if not isinstance(model.get("target"), str):
        print(f"ERROR: Model {i+1} target is not a string: {model.get('target')}")

print("Validation complete.")
