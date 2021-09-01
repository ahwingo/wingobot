# This script can be used to launch the docker container for running the wingobot code.
docker rm wingobot_dev
docker run --gpus all -it --name wingobot_dev \
	   --mount type=bind,source="$(pwd)",target=/wingobot \
	   --env="DISPLAY" \
	   --env="QT_X11_NO_MITSHM=1" \
	   --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
	   wingobot_dev:nvidia bash
export containerId=$(docker ps -l -q)
xhost +local:`docker inspect --format='{{ .Config.Hostname }}' $containerId`
docker start $containerId
docker exec -it wingobot_dev bash

