FROM nvcr.io/nvidia/tensorrt:24.02-py3

# Install required packages for building and running the sample project
RUN apt-get -y update \
&& DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata \
&& apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    pkg-config \
    libavcodec-dev \
    libhdf5-dev \
    libavformat-dev \
    libavdevice-dev \
    libcurl4-openssl-dev \
    libssl-dev

RUN apt-get install -y \
    libopencv-dev

ENV SPDLOGGER_VERSION 1.12.0
RUN wget -P /tmp https://github.com/gabime/spdlog/archive/v${SPDLOGGER_VERSION}.tar.gz
RUN tar -C /tmp -xzf /tmp/v${SPDLOGGER_VERSION}.tar.gz
RUN cp -R /tmp/spdlog-${SPDLOGGER_VERSION}/include/spdlog /usr/include/
RUN rm /tmp/v${SPDLOGGER_VERSION}.tar.gz

RUN rm -rf /workspace
WORKDIR /app

ENV NVIDIA_VISIBLE_DEVICES=all
ENV OPENCV_FFMPEG_LOGLEVEL=0

CMD [ "sleep", "infinity"]