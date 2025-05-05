import os
import shutil

source_dir = r"C:\Users\pcgsa\Downloads\annotated-satellites\raw-images"
target_dir = "test_images"

os.makedirs(target_dir, exist_ok=True)

for root, dirs, files in os.walk(source_dir):
    for filename in files:
        fname = filename.lower()
        if "train" not in fname and "test" not in fname:
            src = os.path.join(root, filename)
            dst = os.path.join(target_dir, filename)
            shutil.copy2(src, dst)
