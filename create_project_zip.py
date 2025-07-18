
import os
import zipfile
import time

def create_project_zip():
    """Create a zip file containing all project files"""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    zip_filename = f"project_backup_{timestamp}.zip"
    
    # Files and directories to exclude
    exclude = [
        '.git', '__pycache__', '.upm', '.config', '.cache',
        zip_filename, 'create_project_zip.py'
    ]
    
    print(f"Creating zip file: {zip_filename}")
    
    # Create the zip file
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Walk through all directories
        for root, dirs, files in os.walk('.'):
            # Remove excluded directories from the walk
            dirs[:] = [d for d in dirs if d not in exclude]
            
            for file in files:
                file_path = os.path.join(root, file)
                
                # Skip the file if it's in the exclude list or is the zip we're creating
                if file in exclude or file_path.startswith('./') and file_path[2:] in exclude:
                    continue
                
                print(f"Adding: {file_path}")
                zipf.write(file_path)
    
    # Get the file size in MB
    size_mb = os.path.getsize(zip_filename) / (1024 * 1024)
    
    print(f"\nZip file created: {zip_filename}")
    print(f"Size: {size_mb:.2f} MB")
    print("\nYou can download this file from the Files panel in Replit.")
    print("Right-click on the file and select 'Download' to save it to your computer.")

if __name__ == "__main__":
    create_project_zip()
