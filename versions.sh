update-alternatives --list python
sudo apt install python3.12
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1
sudo update-alternatives --config python

python -m pip --version
