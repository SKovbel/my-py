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
	# make log - see log files
	#
	# make backup_db - create backup
	# make recovery_db - recovery database
	#
	# make populate - create hr schema
	#

bash:
	/bin/bash bin/bash.sh

start:
	/bin/bash bin/start.sh

rman:
	/bin/bash bin/rman.sh

sqlplus:
	echo $(ARGS)
	/bin/bash bin/sqlplus.sh $(ARGS)

backup:
	/bin/bash bin/backup.sh

log:
	/bin/bash bin/log.sh

populate:
    # hr
    # users
    # temp
    # oracle
    # /tmp/1.log
    # localhost:1521/xepdb1
	/bin/bash bin/populate.sh

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
