#exit
yes | docker system prune
yes | docker system prune -a
yes | docker system prune --volumes
yes | docker container ls -a
yes | docker container prune
yes | docker container stop $(docker container ls -aq)
yes | docker container rm $(docker container ls -aq)
yes | docker image ls
yes | docker image prune
yes | docker image prune -a
yes | docker volume ls
yes | docker volume prune
yes | docker network prune
yes | docker network prune -a
yes | docker network ls
