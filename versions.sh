# https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/
update-alternatives --list python
sudo apt install python3.12
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1
sudo update-alternatives --config python

python -m pip --version
sudo apt autoremove



sudo apt install python3.8 python3.8-distutils
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1
python -m pip install -r requrement.txt

sudo apt install python3.9-pip 
sudo apt install python3.9 python3.9-distutils python3.9-venv
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1
python -m pip install -r requrement.txt


wget https://repo.anaconda.com/archive/Anaconda3-2023.07-2-Linux-x86_64.sh


#2
conda create -n torch python=3.9 anaconda --file /home/work/python/_tourch/requirement.txt
conda info -e
conda install -n yourenvname


conda activate torch
conda deactivate
conda install -c conda-forge google-colab
