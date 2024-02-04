SET ECHO OFF
SET VERIFY OFF

DEFINE pass     = 'oracle'
DEFINE tbs      = 'users'
DEFINE ttbs     = 'temp'
DEFINE pass_sys = 'oracle'
DEFINE log_path = '/tmp/hr-install.log'
DEFINE connect_string = 'localhost:1521/xepdb1'
PROMPT

DEFINE spool_file = &log_path.hr_main.log
SPOOL &spool_file

DROP USER hr CASCADE;
CREATE USER hr IDENTIFIED BY &pass;
ALTER USER hr DEFAULT TABLESPACE &tbs QUOTA UNLIMITED ON &tbs;
ALTER USER hr TEMPORARY TABLESPACE &ttbs;
GRANT CREATE SESSION, CREATE VIEW, ALTER SESSION, CREATE SEQUENCE TO hr;
GRANT CREATE SYNONYM, CREATE DATABASE LINK, RESOURCE , UNLIMITED TABLESPACE TO hr;
 
-- CONNECT sys/&pass_sys AS SYSDBA;
-- GRANT execute ON sys.dbms_stats TO hr;

CONNECT hr/&pass@&connect_string
ALTER SESSION SET NLS_LANGUAGE=American;
ALTER SESSION SET NLS_TERRITORY=America;

@/opt/scripts/human_resources/hr_cre
@/opt/scripts/human_resources/hr_popul
@/opt/scripts/human_resources/hr_idx
@/opt/scripts/human_resources/hr_code
@/opt/scripts/human_resources/hr_comnt
-- @/opt/scripts/human_resources/hr_analz
@/opt/scripts/human_resources/hr_packages
@/opt/scripts/human_resources/hr_proc_func
spool off

