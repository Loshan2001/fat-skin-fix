
import sys
import os
from pathlib import Path

# Add the current directory to sys.path
sys.path.append(os.getcwd())

try:
    print("Attempting to import handler...")
    import handler
    print("Import successful!")
    
    print(f"PWD: {handler.PWD}")
    print(f"Dockerfile path: {handler.dockerfile_path}")
    
    if os.path.exists(handler.dockerfile_path):
        print("Dockerfile exists.")
    else:
        print("Dockerfile DOES NOT exist.")
        
    print("Instantiating App...")
    app = handler.SkinFixApp()
    print("App instantiated successfuly.")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
