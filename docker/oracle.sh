mkdir -p ./../tmp/oracle/scripts
mkdir -p ./../tmp/oracle/dump

sudo chmod 666 /var/run/docker.sock

docker login container-registry.oracle.com
docker-compose -f cnf/oracle.yml up
