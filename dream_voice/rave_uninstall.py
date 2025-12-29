import torch
import shutil
import os

# 1. Get the directory where torch stores these downloaded repos
hub_dir = torch.hub.get_dir() 
print(f"📍 Torch Hub Cache found at: {hub_dir}")

# 2. Look for any folder related to RAVE or acids-ircam
found = False
if os.path.exists(hub_dir):
    for item in os.listdir(hub_dir):
        # Identify folders matching the RAVE repo pattern
        if "acids-ircam" in item or "rave" in item.lower():
            full_path = os.path.join(hub_dir, item)
            print(f"🗑️  Deleting corrupted folder: {full_path}")
            
            # Force delete the directory tree
            try:
                shutil.rmtree(full_path)
                found = True
            except PermissionError:
                print(f"❌ Permission Error: Close any other Python windows and try again.")

if not found:
    print("✅ No RAVE cache files found. You are clean.")
else:
    print("✅ Cleanup complete.")