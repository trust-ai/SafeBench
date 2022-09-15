FROM gilgameshd/platform

USER root

COPY xorg.conf /etc/X11/xorg.conf
COPY nvidia /usr/lib/nvidia
COPY xorg /usr/lib/xorg/modules
COPY gnu_nvidia /usr/lib/x86_64-linux-gnu/nvidia/
COPY requirements.txt /home/carla/requirements.txt
COPY sensor.py /home/carla/carla-ros-bridge/catkin_ws/src/ros-bridge/carla_ros_bridge/src/carla_ros_bridge/sensor.py

RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - \
    && rm /etc/apt/sources.list.d/cuda.list \
    && rm /etc/apt/sources.list.d/nvidia-ml.list \
    && apt-get update \
    && apt-get install -y gdm3 \
    && service gdm3 stop \
    && mkdir VirtualGL \
    && cd VirtualGL \
    && wget https://versaweb.dl.sourceforge.net/project/virtualgl/3.0/virtualgl_3.0_amd64.deb \
    && apt-get update \
    && apt-get install -y libxtst6 libxv1 libegl1-mesa \
    && dpkg -i virtualgl*.deb \
    && apt install -f \
    && apt-get install -y x11-utils \
    && cd .. \
    && mkdir TurboVNC \
    && cd TurboVNC \
    && apt-get install -y libxmuu1 xauth \
    && wget https://udomain.dl.sourceforge.net/project/turbovnc/2.2.7/turbovnc_2.2.7_amd64.deb \
    && dpkg -i turbovnc*.deb \
    && apt install -f \
    && apt-get install -y xauth \
    && apt-get install -y lsof \
    && apt-get install mesa-utils \
    && mkdir -p /run/dbus \
    && service dbus start \
    && rm /var/run/dbus/pid \
    && dbus-daemon --system \
    && apt-get install -y xserver-xorg-video-fbdev \
    && apt-get install -y xserver-xorg-video-vesa \
    && apt-get install -y xserver-xorg-video-nouveau \
    && pip3 install -r /home/carla/requirements.txt \
    && apt-get install tcl-dev tk-dev python3-tk \
    && rm -f /home/carla/requirements.txt

USER carla

RUN pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html \
    && pip3 install torch-scatter -f https://data.pyg.org/whl/torch-1.9.0+cu111.html \
    && pip3 install protobuf==3.20
