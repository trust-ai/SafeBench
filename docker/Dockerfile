FROM carlasim/carla:0.9.13

ARG DEBIAN_FRONTEND=noninteractive

USER root

RUN apt-get install -y --no-install-recommends apt-utils \
    && apt-get install -y wget \
    && sed -i '/developer\.download\.nvidia\.com\/compute\/cuda\/repos/d' /etc/apt/sources.list \
    && sed -i '/developer\.download\.nvidia\.com\/compute\/cuda\/repos/d' /etc/apt/sources.list.d/* \
    && sed -i '/developer\.download\.nvidia\.com\/compute\/machine-learning\/repos/d' /etc/apt/sources.list.d/* \
    && apt-key del 7fa2af80 \
    && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb \
    && dpkg -i cuda-keyring_1.0-1_all.deb \
    && rm -f cuda-keyring_1.0-1_all.deb

RUN apt-get update \
    && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get install -y sudo htop tmux psmisc python3.7 python3.7-distutils python3.7-dev python3-pip git wget unzip g++ cmake tar \
    && apt-get install -y libpng16-16 libjpeg-turbo8 libtiff5 libomp5 \
    && apt-get install -y libice6 libsm6 libxaw7 libxkbfile1 libxmu6 libxpm4 libxt6 x11-common x11-xkb-utils xkb-data \
    && ln -s /usr/bin/python3.7 /usr/bin/python

RUN wget https://phoenixnap.dl.sourceforge.net/project/turbovnc/3.0.3/turbovnc_3.0.3_amd64.deb \
    && dpkg -i turbovnc*.deb \
    && rm -f turbovnc*.deb

RUN useradd -ms /bin/bash safebench \
    && usermod -aG sudo safebench \
    && echo "safebench:safe" | chpasswd

USER safebench
WORKDIR /home/safebench
ENV PATH="${PATH}:/home/safebench/.local/bin"

RUN python -m pip install --upgrade pip \
    && pip install gdown \
    && gdown https://drive.google.com/uc?id=139vLRgXP90Zk6Q_du9cRdOLx7GJIw_0v \
    && mkdir carla \
    && tar -zxf CARLA*.tar.gz -C carla \
    && rm -f CARLA*.tar.gz

RUN echo "export CARLA_ROOT=/home/safebench/carla" >> ~/.bashrc \
    && echo "export PYTHONPATH=\$PYTHONPATH:\${CARLA_ROOT}/PythonAPI_docker/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg" >> ~/.bashrc \
    && echo "export PYTHONPATH=\$PYTHONPATH:\${CARLA_ROOT}/PythonAPI_docker/carla/agents" >> ~/.bashrc \
    && echo "export PYTHONPATH=\$PYTHONPATH:\${CARLA_ROOT}/PythonAPI_docker/carla" >> ~/.bashrc \
    && echo "export PYTHONPATH=\$PYTHONPATH:\${CARLA_ROOT}/PythonAPI_docker" >> ~/.bashrc

RUN git clone https://github.com/trust-ai/SafeBench.git \
    && cd SafeBench \
    && pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117  \
    && pip install -r requirements.txt \
    && pip install moviepy \
    && pip install -e .