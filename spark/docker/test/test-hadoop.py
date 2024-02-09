from hdfs import InsecureClient

client = InsecureClient('http://127.17.0.1:9870', user='superuser')

# Define paths
local_file_path = './Makefile'
hdfs_dir_path = '/test-1'
hdfs_file_path = '/test-1/Makefile2'

if not client.status(hdfs_dir_path, strict=False):
    client.makedirs(hdfs_dir_path)

directories = client.list('/')
print("Directories in HDFS:", directories)

with open(local_file_path, 'rb') as local_file:
   client.upload(hdfs_file_path, local_file_path)

client.disconnect()
