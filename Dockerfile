FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime
SHELL ["/bin/bash", "-c"]

RUN apt-get update && \
    apt-get install -y swig libsndfile1-dev libasound2-dev && \
    apt-get clean

WORKDIR /app

# install requirements
COPY requirements.txt /app/
RUN pip install -r <(cat requirements.txt | grep -v 'torch==1.7.0')
