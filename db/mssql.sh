#!/bin/sh

mkdir -p ./../tmp/mssql/data
#sudo chown 10001:10001 -R  ./../tmp/mssql/data
sudo chmod 777 -R  ./../tmp/mssql/data

python3.11 -m venv ../tmp/db-venv
source ../tmp/db-venv/bin/activate
pip install docker-compose

docker-compose -f cnf/mssql.yml up

exit
#docker run -e "ACCEPT_EULA=Y" -e "MSSQL_SA_PASSWORD=superadmin" \
#   -p 1433:1433 --name sql1 --hostname sql1 \
#   -d \
#   mcr.microsoft.com/mssql/server:2022-latest
