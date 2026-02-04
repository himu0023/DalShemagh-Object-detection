# diagnostic.py
import os

# Path to your data
data_path = "data/raw/dal_shemagh"
full_path = os.path.abspath(data_path)

print("="*60)
print("DIAGNOSTIC CHECK")
print("="*60)
print(f"Looking in: {full_path}")
print(f"Folder exists: {os.path.exists(full_path)}")

if os.path.exists(full_path):
    print("\nContents of dal_shemagh folder:")
    print("-" * 40)
    for item in os.listdir(full_path):
        item_path = os.path.join(full_path, item)
        print(f"{item} - {'DIR' if os.path.isdir(item_path) else 'FILE'}")
    
    # Check images folder
    images_path = os.path.join(full_path, "images")
    print(f"\nImages folder: {images_path}")
    print(f"Exists: {os.path.exists(images_path)}")
    
    if os.path.exists(images_path):
        print(f"\nContents of images folder:")
        print("-" * 40)
        image_files = os.listdir(images_path)
        if image_files:
            for img in image_files[:10]:  # Show first 10 files
                print(f"  {img}")
            if len(image_files) > 10:
                print(f"  ... and {len(image_files)-10} more files")
        else:
            print("  EMPTY - No files found!")
        
        # Check file extensions
        print(f"\nFile extensions in images folder:")
        extensions = {}
        for file in image_files:
            ext = os.path.splitext(file)[1].lower()
            extensions[ext] = extensions.get(ext, 0) + 1
        
        for ext, count in extensions.items():
            print(f"  {ext}: {count} files")
            
    # Check labels folder
    labels_path = os.path.join(full_path, "labels")
    print(f"\nLabels folder: {labels_path}")
    print(f"Exists: {os.path.exists(labels_path)}")
    
    if os.path.exists(labels_path):
        print(f"\nContents of labels folder (first 10):")
        print("-" * 40)
        label_files = os.listdir(labels_path)
        if label_files:
            for lbl in label_files[:10]:
                print(f"  {lbl}")
            if len(label_files) > 10:
                print(f"  ... and {len(label_files)-10} more files")
        else:
            print("  EMPTY - No files found!")
print("="*60)