# FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04
FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
	python3-opencv \
	ca-certificates \
	python3-dev \
	python3-pip \
	git \
	wget \
	sudo \
	ninja-build \
	vim
RUN ln -sv /usr/bin/python3 /usr/bin/python

COPY . /home/appuser
WORKDIR /home/appuser

# https://github.com/facebookresearch/detectron2/issues/3933
ENV PATH="/home/appuser/.local/bin:${PATH}"

# install dependencies
# See https://pytorch.org/ for other options if you use a different version of CUDA
# RUN pip install torch==1.10 torchvision==0.11.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
RUN pip install --user torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN pip install setuptools==59.5.0
RUN pip install -r requirements.txt
RUN python setup.py develop
