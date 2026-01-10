FROM nvidia/cuda:12.3.0-devel-ubuntu22.04
#FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    git 

RUN apt-get install -y libopencv-dev

WORKDIR /src

COPY . .

RUN ./compile_main.sh

ENTRYPOINT [ "/src/main" ]

