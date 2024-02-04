#!/bin/sh
#
source _source.sh

if [[ 1 -eq 0 ]]; then
    SHUTDOWN IMMEDIATE;
    STARTUP MOUNT;
    ALTER DATABASE ARCHIVELOG;
    ALTER DATABASE OPEN;
    SHOW PARAMETER LOG_ARCHIVE_START;
    SHOW PARAMETER LOG_ARCHIVE_DEST;
fi

# Docker dirs, file, format
DDIR_BACKUP=/opt/backup
DDIR_LOG=/opt/backup/log
LOG_FILE="$DDIR_LOG/rman_backup_$(date +'%Y%m%d_%H%M%S').log"
DATE_FORMAT='YYYY-MM-DD:HH24:MI:SS'

# Rman script, part of docker script
RMAN_SCRIPT="
RUN {
    SQL 'ALTER SYSTEM SWITCH LOGFILE';

    ALLOCATE CHANNEL c1 DEVICE TYPE DISK FORMAT '$DDIR_BACKUP/%U';
    CONFIGURE RETENTION POLICY TO RECOVERY WINDOW OF 3 DAYS;

    BACKUP DATABASE;
    BACKUP ARCHIVELOG ALL;
    BACKUP CURRENT CONTROLFILE;

    DELETE BACKUP COMPLETED BEFORE 'SYSDATE-0';
    CROSSCHECK ARCHIVELOG ALL;
    DELETE EXPIRED ARCHIVELOG ALL;

    # BACKUP DATABASE PLUS ARCHIVELOG;
    # DELETE BACKUP TYPE 'FULL' COMPLETED BEFORE 'SYSDATE-7' KEEP 3;
}
"

# Docker script, contains rman script
DOCKER_SCRIPT="
echo '>>>>>>>>>>>>>> RMAN script >>>>>>>>>>>>>>'

export RMAN_LOG='$LOG_FILE'
export NLS_DATE_FORMAT='$DATE_FORMAT'

mkdir -p $DDIR_BACKUP
mkdir -p $DDIR_LOG
ls -1t $DDIR_BACKUP | head -10
echo 'Backup dir=$DDIR_BACKUP'

rman target / nocatalog <<EOF
    $RMAN_SCRIPT
EOF

# cat $LOG_FILE
ls -1t $DDIR_BACKUP | head -5 | sort
"

# Call script
docker exec -it oracle_db_1 sh -c "$DOCKER_SCRIPT"



#
# BEGIN
#   DBMS_SCHEDULER.create_job (
#      job_name        => 'RMAN_BACKUP_JOB',
#      job_type        => 'EXECUTABLE',
#      job_action      => '/path/to/backup_script.sh',
#      start_date      => SYSTIMESTAMP,
#      repeat_interval => 'FREQ=DAILY; BYHOUR=2', -- Adjust the frequency as needed
#      enabled         => TRUE
#   );
# END;
