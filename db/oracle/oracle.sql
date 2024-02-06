
SHUTDOWN IMMEDIATE;
STARTUP MOUNT;
ALTER DATABASE ARCHIVELOG;
ARCHIVE LOG LIST;
ALTER DATABASE OPEN;

create or replace context t1_ctx using t1_pkg accessed globally;


SELECT 
  sys_context('USERENV', 'SESSIONID') as session_id,
  sys_context('USERENV', 'SESSION_USER') as session_user,
  sys_context('USERENV', 'HOST') as host,
  sys_context('USERENV', 'IP_ADDRESS') as ip_address,
  sys_context('USERENV', 'OS_USER') as os_user,
  sys_context('USERENV', 'CURRENT_SCHEMA') as current_schema
FROM dual;sh




SELECT * FROM DBA_TABLESPACES;
SELECT * FROM DBA_DATA_FILES;

CREATE TABLESPACE EXAMPLE_TABLESPACE
  DATAFILE '/opt/oracle/oradata/XE/custom.dbf' SIZE 100M
  AUTOEXTEND ON NEXT 10M
  MAXSIZE UNLIMITED;

ALTER TABLESPACE EXAMPLE_TABLESPACE ADD DATAFILE '/opt/oracle/oradata/XE/custom01.dbf' SIZE 100M AUTOEXTEND ON NEXT 10M MAXSIZE UNLIMITED;
