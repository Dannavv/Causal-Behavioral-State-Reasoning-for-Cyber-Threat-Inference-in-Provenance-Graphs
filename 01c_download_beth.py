import kagglehub

def download_beth():
    print("Downloading BETH dataset from Kaggle...")
    # Downloads to the default kaggle cache, returns the path
    path = kagglehub.dataset_download("joshuajung/beth-dataset")
    print(f"BETH Dataset downloaded to {path}")
    
    # We will write a file to indicate the path for the preprocessor
    with open("data/raw/beth_path.txt", "w") as f:
        f.write(path)

if __name__ == "__main__":
    download_beth()
