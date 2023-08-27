FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04

# avoid tzdata interaction
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt update && \
    apt install -y libsndfile1-dev \
                   libgl1 \
                   python3.8 \
                   python3-pip \
                   python3.8-venv && \
    rm -rf /var/lib/apt/lists

# https://github.com/NVIDIA/nvidia-docker/issues/1009#issuecomment-1181312052
RUN rm /etc/apt/sources.list.d/cuda.list

# install pytorch and cuda
ARG PYTORCH='2.0.1'
ARG VERSION='torch=='$PYTORCH'.*'
# Example: `cu102`, `cu113`, 'nightly/cu121', etc. ctransformers needs cuda 12+
ARG CUDA='nightly/cu121'
RUN python3 -m pip install --no-cache-dir -U $VERSION --extra-index-url https://download.pytorch.org/whl/$CUDA

# avoid gunicorn not found errors
ENV PYTHONPATH=/app 

RUN pip3 install uvicorn[standard]==0.20.0 gunicorn==20.1.0 fastapi[all]==0.88.0


# install dependencies
RUN python3 -m pip install --upgrade pip
RUN pip3 install invisible_watermark transformers accelerate safetensors diffusers --upgrade

COPY ./app /app

WORKDIR /app

EXPOSE 7080

# Start the app
CMD ["python3", "-m", "gunicorn", "-b", "0.0.0.0:7080", "main:app","--workers","2","--timeout","180","-k","uvicorn.workers.UvicornWorker"]
