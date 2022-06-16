#pip3 install -r ./src/requirements.txt
catkin_make
. ./devel/setup.sh
roslaunch manager manager.launch