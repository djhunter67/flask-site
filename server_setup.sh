#!/usr/bin/env bash

sudo apt update
sudo apt upgrade -y

# Install some OS dependencies
sudo apt install -y -q build-essential git unzip zip nload tree python3-pip python3-dev python3-venv nginx

# gzip support for uwgsi
sudo apt install -y --no-install-recommends -q libpcre3-dev libz-dev

# Stop the hackers
sudo apt install -y fail2ban

sudo ufw allow 22
sudo ufw allow 80
sudo ufw allow 443
sudo ufw enable

# Basic git setup
git config --global credential.helper cache
git config --global credential.helper 'cache --timeout=720000'

# Basic git profile setup
git config --global user.email "djhunter67@gmail.com"
git config --global user.name "Christerpher Hunter"

# Web app file structure
sudo mkdir /apps
sudo chmod 777 /apps
mkdir -p /apps/logs/personal_site/app_log.log
cd /apps

# Create vitrual env for the app
cd /apps
python3 -m venv venv
source /apps/venv/bin/activate
pip install -U pip setuptools httpie glances uwsgi

# Clone the repo
cd /apps
git clone http://192.168.110.24:9080/djhunter67/flask_site.git

# Setup the web app
cd /apps/flask_site/
pip install -r requirements.txt

# Copy and enable the deamon
sudo cp /apps/flask_site/main.service /etc/systemd/system/

sudo systemctl start main
sudo systemctl status main
sudo systemctl enable main

# Setup the public facinf server
sudo apt install -y nginx

# rm default nginx page
sudo rm /etc/nginx/sites-enabled/default

# Place nginx config accordingly
sudo cp /apps/flask_site/main.nginx /etc/nginx/sites-enabled/main.nginx
sudo update-rc.d nginx enable
sudo service nginx restart

# Setup SSL support via Let's Encrypt:
sudo add-apt-repository ppa:certbot/certbot
sudo apt install -y python3-certbot-nginx
sudo certbot --nginx -d christerpher.com
