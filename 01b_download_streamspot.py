import urllib.request
import os
import tarfile

def download_streamspot():
    os.makedirs("data/raw/streamspot", exist_ok=True)
    url = "https://raw.githubusercontent.com/sbustreamspot/sbustreamspot-data/master/all.tar.gz"
    dest = "data/raw/streamspot/all.tar.gz"
    if not os.path.exists(dest):
        print(f"Downloading {url} to {dest}...")
        urllib.request.urlretrieve(url, dest)
        print("Download complete.")
    else:
        print(f"{dest} already exists.")
    
    print("Extracting...")
    with tarfile.open(dest, "r:gz") as tar:
        tar.extractall(path="data/raw/streamspot")
    print("Extraction complete.")

if __name__ == "__main__":
    download_streamspot()
