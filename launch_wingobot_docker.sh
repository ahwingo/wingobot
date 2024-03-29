docker run --rm --gpus all -it --name wingobot_dev_visual --env="DISPLAY" \
	      --volume="/etc/group:/etc/group:ro" \
	         --volume="/etc/passwd:/etc/passwd:ro" \
		    --volume="/etc/shadow:/etc/shadow:ro" \
		       --volume="/etc/sudoers.d:/etc/sudoers.d:ro" \
		          --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
						     --mount type=bind,source="$(pwd)",target=/wingobot wingobot_dev:latest bash
