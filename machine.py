import os
import json
import uuid
import shutil
from datetime import datetime

# Define the file path
STORAGE_FILE = os.path.expanduser(r"~\AppData\Roaming\Cursor\User\globalStorage\storage.json")

# Generate a random ID
def generate_random_id():
    return uuid.uuid4().hex

# Get new ID
NEW_ID = generate_random_id()

# Backup the file
def backup_file(file_path):
    if os.path.exists(file_path):
        backup_path = f"{file_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy(file_path, backup_path)
        print(f"Backup created: {backup_path}")

# Ensure the directory exists
os.makedirs(os.path.dirname(STORAGE_FILE), exist_ok=True)

# Backup the file if it exists
backup_file(STORAGE_FILE)

# Create a new JSON file if it doesn't exist
if not os.path.exists(STORAGE_FILE):
    with open(STORAGE_FILE, "w") as f:
        json.dump({}, f)

# Update the machineId
with open(STORAGE_FILE, "r") as f:
    data = json.load(f)

data["telemetry.machineId"] = NEW_ID

with open(STORAGE_FILE, "w") as f:
    json.dump(data, f, indent=4)

print(f"Successfully updated machineId to: {NEW_ID}")
input("Enter")