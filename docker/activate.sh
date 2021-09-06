# This script can be used to launch the docker container for running the wingobot code.
docker rm wingobot_dev
docker run --gpus all -it --name wingobot_dev --mount type=bind,source="$(pwd)",target=/wingobot wingobot_dev:nvidia bash

