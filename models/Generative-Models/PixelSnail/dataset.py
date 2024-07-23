import os
import zipfile
import urllib.request

url = 'https://storage.googleapis.com/kaggle-data-sets/423204/1026106/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20240714%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240714T164714Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=03f13ff33521f451958f87fa965511834a40c26d0e126c36ca6f5088549b6ea0ac51054db08180d200b9d38c3b63094d4bad680175b5df19a868802f21a30a03fbd9c6b808cd873caa7059e67169a1f0cfa953ab4660cb3b494f58f6a9594896468f902825387a573cf9c47e422b1ae1f8c749199271e323276e2540192dbcd448ca2eeb45a2130f19ffc8fb327a4bb200fe263008d3f479f0ed7598830a587be67eabb9de32b04c2d3e3421523f5d34fccb0755c6f8d1be6b5076fa1725fba8ac071d8036c267ffdfef03cd5edbd6d00e759b61332fb3036e43e070ee7736afcc0f145c00e1ab95aa53ccaa584dc099c2282a20a231b59d59615472bfdba7fd'
dir = f'{os.path.dirname(os.path.abspath(__file__))}/../../tmp/gems'
zip = f'{dir}/gems.zip'

os.makedirs(dir, exist_ok=True)
urllib.request.urlretrieve(url, zip)
with zipfile.ZipFile(zip, 'r') as zip_file:
    zip_file.extractall(dir)
