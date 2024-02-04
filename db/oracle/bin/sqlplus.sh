#!/bin/sh
echo $1
docker exec -it oracle_db_1 sqlplus / as sysdba 