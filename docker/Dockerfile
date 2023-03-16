FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt install -y sudo htop tmux psmisc python3 python3-pip python-is-python3 git wget unzip g++ cmake tar \
    && apt-get install -y libpng16-16 libjpeg-turbo8 libtiff5 libomp5 \
    && apt-get install -y libice6 libsm6 libxaw7 libxkbfile1 libxmu6 libxpm4 libxt6 x11-common x11-xkb-utils xkb-data
#    && apt install -y htop tmux psmisc python3.7 python3.7-distutils python3-pip python-is-python3 git wget unzip g++ cmake

RUN wget https://phoenixnap.dl.sourceforge.net/project/turbovnc/3.0.3/turbovnc_3.0.3_amd64.deb \
    && dpkg -i turbovnc*.deb \
    && rm -f turbovnc*.deb

RUN useradd -ms /bin/bash safebench \
    && usermod -aG sudo safebench \
    && echo "safebench:safe" | chpasswd

USER safebench
WORKDIR /home/safebench
ENV PATH="${PATH}:/home/safebench/.local/bin"

RUN pip install gdown \
    && gdown https://drive.google.com/uc?id=1A4z3RKXqVYpOmsEZkPBV1Pbw3B8aeSMp \
    && mkdir carla \
    && tar -zxf CARLA*.tar.gz -C carla \
    && rm -f CARLA*.tar.gz

RUN echo "export CARLA_ROOT=/home/safebench/carla" >> ~/.bashrc \
    && echo "export PYTHONPATH=\$PYTHONPATH:\${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.13-py3.8-linux-x86_64.egg" >> ~/.bashrc \
    && echo "export PYTHONPATH=\$PYTHONPATH:\${CARLA_ROOT}/PythonAPI/carla/agents" >> ~/.bashrc \
    && echo "export PYTHONPATH=\$PYTHONPATH:\${CARLA_ROOT}/PythonAPI/carla" >> ~/.bashrc \
    && echo "export PYTHONPATH=\$PYTHONPATH:\${CARLA_ROOT}/PythonAPI" >> ~/.bashrc

RUN git clone https://github.com/trust-ai/SafeBench.git \
    && cd SafeBench \
    && pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117  \
    && pip install -r requirements.txt \
    && pip install -e .