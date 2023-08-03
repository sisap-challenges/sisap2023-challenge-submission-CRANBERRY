# Define base image/operating system
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install software
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl ca-certificates git
# RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates
RUN curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda3 -b
RUN rm -f Miniconda3-latest-Linux-x86_64.sh

RUN curl -O https://download.oracle.com/java/17/latest/jdk-17_linux-x64_bin.tar.gz
RUN tar xvfz jdk-17_linux-x64_bin.tar.gz
RUN rm -f jdk-17_linux-x64_bin.tar.gz

RUN curl -O https://dlcdn.apache.org/maven/maven-3/3.9.3/binaries/apache-maven-3.9.3-bin.tar.gz
RUN tar xvfz apache-maven-3.9.3-bin.tar.gz
RUN rm -f apache-maven-3.9.3-bin.tar.gz

# Copy files and directory structure to working directory
COPY . . 
#COPY bashrc ~/.bashrc
# this will not work for other 17 minor versions, TODO determine extracted dire
ENV PATH=/miniconda3/bin:/jdk-17.0.8/bin:/apache-maven-3.9.3/bin/:${PATH}

SHELL ["/bin/bash", "--login", "-c"]
RUN conda init bash
#RUN echo 'export PATH=/miniconda3/bin:$PATH' > ~/.bashrc

RUN conda create -n CRANBERRY  python=3.11
RUN conda run -n CRANBERRY pip install -r requirements.txt
RUN bash sisap23-setup.sh
RUN mkdir -p /Similarity_search/Dataset/Dataset 
RUN mkdir -p /Similarity_search/Dataset/Query
RUN ln -s /data/clip768v2/300K/dataset.h5 /Similarity_search/Dataset/Dataset/laion2B-en-clip768v2-n=300K.h5
RUN ln -s /data/clip768v2/10M/dataset.h5 /Similarity_search/Dataset/Dataset/laion2B-en-clip768v2-n=10M.h5
RUN ln -s /data/clip768v2/30M/dataset.h5 /Similarity_search/Dataset/Dataset/laion2B-en-clip768v2-n=30M.h5
RUN ln -s /data/clip768v2/100M/dataset.h5 /Similarity_search/Dataset/Dataset/laion2B-en-clip768v2-n=100M.h5
RUN ln -s /data/clip768v2/100M/query.h5 /Similarity_search/Dataset/Query/private-queries-gold-10k-clip768v2.h5
# Set container's working directory - this is arbitrary but needs to be "the same" as the one you later use to transfer files out of the docker image
#WORKDIR result


# Run commands specified in "run.sh" to get started

#ENTRYPOINT [ "/bin/bash", "-l", "-c" ]
ENTRYPOINT [ "/bin/bash", "/sisap23-run.sh"]
