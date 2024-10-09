import os
import zipfile
import urllib.request

url = 'https://storage.googleapis.com/kaggle-data-sets/423204/1026106/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20240723%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240723T214510Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=39447d0acecaee97feb433e9d5fb32b2a49198e76134555ab2b7c042ce91d9fe5e1ae9a5095cd1765fc0952241e521013f1ee9ba1a1066bcd702b2dcc5b8f81cfb19435ab5d2435a5ab5090767e943e73e3943364a77d0bb76ab68c8d98198e172966bfa108328d96ac7d4b1ef24cdcc03e6be946b7d9ad9b29324b4586d41bd549db065d3a3fe24729a3b6392f27ac8bf523a42b1ea2744e22513b0277383a1cea8eec24430f2cacf8e00b9937a472197ca336a4017e3800d98c5c2ed7927c89a63e8e119c758c5144c942797834259b429c9da76395d54057a259a10815b5c636819d7351ba751026ff823779ed7de286dfe5f1cc8cc7b1ade72676b16353c'
dir = f'{os.path.dirname(os.path.abspath(__file__))}/../../tmp/gems'
zip = f'{dir}/gems.zip'

os.makedirs(dir, exist_ok=True)
urllib.request.urlretrieve(url, zip)
with zipfile.ZipFile(zip, 'r') as zip_file:
    zip_file.extractall(dir)
