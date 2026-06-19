import kagglehub
import os

print("Downloading OpTC benign dataset...")
path = kagglehub.dataset_download("faihaj/optc-corrected-benign-dataset-sep16-parquet")
print("Download complete. Dataset path:")
print(path)

# Let's list the files to understand the structure
print("\nFiles in the downloaded dataset:")
for root, dirs, files in os.walk(path):
    for file in files:
        print(os.path.join(root, file))
