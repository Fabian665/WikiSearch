# Echo commands
set -v
sudo apt-get update
sudo apt-get install -yq git python3 python3-setuptools python3-dev build-essential tmux
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
sudo python3 get-pip.py
sudo pip3 install --no-input nltk==3.6.3 Flask==2.0.2 --no-cache-dir flask-restful==0.3.9 numpy==1.21.4 google-cloud-storage==1.43.0 pandas werkzeug==2.2.2 numpy gensim
# python -m nltk.downloader stopwords
cd ~
git clone https://github.com/Fabian665/WikiSearch
tmux new-session -d -s my_session 'python3 WikiSearch/search_frontend.py'
