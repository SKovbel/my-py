#!/bin/sh

docker exec -it oracle_db_1 sqlplus system/oracle@localhost:1521/xepdb1 @/opt/scripts/human_resources/hr_main.sql
