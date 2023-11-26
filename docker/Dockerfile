###################################################### Image ######################################################
FROM ubuntu:22.04

################################################## Date and Time ##################################################
ENV TZ="America/Chicago"
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime
RUN echo $TZ > /etc/timezone && rm -rf /var/lib/apt/lists/*

RUN apt-get update -y
RUN apt-get install -y locales curl wget tar sudo git apt-utils
RUN localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8

ENV LANG en_US.utf8

################################################## Dependencies ###################################################
RUN apt-get install -y gcc g++ vim automake make clang build-essential cmake llvm-dev

ADD ./intel.tar.gz /

##################################################### Repo ########################################################
WORKDIR /workspace
