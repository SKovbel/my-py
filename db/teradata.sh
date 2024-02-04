mkdir -p data/terada

#python3.13 -m venv ../tmp/db-venv
#source ../tmp/db-venv/bin/activate
#pip install docker-compose


sudo chmod 666 /var/run/docker.sock

docker login
docker-compose -f cnf/teradata.yml up
