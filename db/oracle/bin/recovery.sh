#!/bin/sh
RDIR=/opt/oracle/backup
DATE=$(date +'%Y%m%d_%H%M%S')

RMAN_SCRIPT="
RUN {
    RESTORE DATABASE PREVIEW;
    RESTORE DATABASE;
    RECOVER DATABASE;
}
"

DOCKER_SCRIPT="
echo '>>>>>>>>>>>>>> RMAN script >>>>>>>>>>>>>>'

export RMAN_LOG='$RDIR/rman_backup_$DATE.log'
export NLS_DATE_FORMAT='YYYY-MM-DD:HH24:MI:SS'

rman target / nocatalog <<EOF
    $RMAN_SCRIPT
EOF
"

docker exec -it oracle_db_1 sh -c "$DOCKER_SCRIPT"