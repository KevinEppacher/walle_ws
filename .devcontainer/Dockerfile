FROM osrf/ros:noetic-desktop-full

# Add vscode user with same UID and GID as your host system
# (copied from https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user#_creating-a-nonroot-user)
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME
# Switch from root to user
USER $USERNAME

# Add user to video group to allow access to webcam
RUN sudo usermod --append --groups video $USERNAME

# Update all packages
RUN sudo apt update && sudo apt upgrade -y

# Install pip
RUN sudo apt-get install -y python3-pip

# Install Python libraries
RUN pip3 install pandas matplotlib

# Install tkinter
RUN sudo apt-get install -y python3-tk

RUN sudo apt install -y ros-noetic-plotjuggler-ros

RUN sudo apt-get install -y ros-noetic-dwa-local-planner

RUN sudo apt-get install -y ros-noetic-gmapping

RUN sudo apt-get update && sudo apt-get install -y ros-noetic-map-server

RUN sudo apt-get install -y ros-noetic-teb-local-planner

RUN sudo apt-get install -y ros-noetic-turtlebot3* ros-noetic-turtlebot3 \
    && echo "export TURTLEBOT3_MODEL=waffle_pi" >> ~/.bashrc

# Install Git
RUN sudo apt update

RUN sudo apt install -y git

RUN sudo apt install -y python3 python3-pip

RUN pip install casadi

RUN pip install numpy scipy

# Rosdep update
RUN rosdep update

# Source the ROS setup file
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc