.MAIN: help
.PHONY: help

help:
	# 
	# Hello!
	#
	# make start - start Oracle container
	#
	# make bash - Oracle container's bash
	# make rman - Oracle container's rman
	# make sqlplus - Oracle container's sqlplus
	# make log - 
	#
	# make backup_db - create backup
	# make recovery_db - recovery database
	#

start:
	/bin/bash bin/start.sh

rman:
	/bin/bash bin/rman.sh

sqlplus:
	/bin/bash bin/sqlplus.sh

backup:
	/bin/bash bin/backup.sh

log:
	/bin/bash bin/log.sh

setup_archive_log:
    # SHUTDOWN IMMEDIATE;
    # STARTUP MOUNT;
    # ALTER DATABASE ARCHIVELOG;
    # ALTER DATABASE OPEN;
    # SHOW PARAMETER LOG_ARCHIVE_START;
    # SHOW PARAMETER LOG_ARCHIVE_DEST;

setup_oem:
    # EXEC dbms_xdb_config.setglobalportenabled(TRUE);
    # EXEC DBMS_XDB.SETHTTPPORT(8080);
    # EXEC DBMS_XDB.SETHTTPPORT(5500);
    # http://localhost:2020/em

recovery_db:
	@read -p "Are you sure you want to continue? (y/n): " answer; \
	if [ "$$answer" != "y" ]; then \
		echo "Aborted."; \
		exit 1; \
	fi
	/bin/bash bin/recovery.sh
