#FROM nvcr.io/nvidia/tensorflow:20.12-tf1-py3
FROM nvcr.io/nvidia/tensorflow:20.12-tf2-py3
RUN apt-get update && apt-get install -y vim
