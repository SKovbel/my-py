sqlite3 label_studio.sqlite3 -header -csv "select file from data_import_fileupload where project_id=3;" > files3.csv
sqlite3 label_studio.sqlite3 -header -csv "select data from task where updated_at < '2024-01-13 13:00';" > date3.csv


SELECT *
FROM task
WHERE datetime(updated_at) < datetime('2024-01-13 13:00:00');

sqlite3 label_studio.sqlite3 -header -csv "select file from data_import_fileupload where project_id=3;" > files3.csv
