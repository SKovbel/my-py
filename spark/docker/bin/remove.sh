#exit
docker system prune
docker system prune -a
docker system prune --volumes
docker container ls -a
docker container prune
docker container stop $(docker container ls -aq)
docker container rm $(docker container ls -aq)
docker image ls
docker image prune
docker image prune -a
docker volume ls
docker volume prune
docker network prune
docker network prune -a
docker network ls
