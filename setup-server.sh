#!/bin/bash

# Should be run in tmux

sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

sudo apt-get update
sudo apt-get install -y htop git python3-pip
sudo pip3 install virtualenv

git clone https://github.com/reo7sp/vk-text-likeness
cd vk-text-likeness
virtualenv .venv
source .venv/bin/activate
pip install -r requirements.txt

python main.py "$@"
