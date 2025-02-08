import os
import zipfile

# Define the folder containing your updated code
folder_path = r"C:\Users\gill_\OneDrive\Documents\tradingbot"

# Define the name for the output ZIP file
zip_filename = "updated_code.zip"
zip_filepath = os.path.join(folder_path, zip_filename)

# Create a ZipFile object in write mode with ZIP_DEFLATED compression
with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
    # Walk through all subdirectories and files in the folder_path
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # Only include .py files (you can modify this if you need additional file types)
            if file.endswith('.py'):
                full_path = os.path.join(root, file)
                # Create a relative archive name so the directory structure is preserved inside the zip file
                archive_name = os.path.relpath(full_path, folder_path)
                zipf.write(full_path, arcname=archive_name)

print(f"All Python scripts have been zipped into: {zip_filepath}")
