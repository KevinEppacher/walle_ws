cmake_minimum_required(VERSION 3.0.2)
project(amr_control)

find_package(catkin REQUIRED COMPONENTS
  dynamic_reconfigure
  nav_msgs
  roscpp
  roslib
  rospy
  sensor_msgs
  std_msgs
  tf
  tf2
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES amr_control
#  CATKIN_DEPENDS dynamic_reconfigure nav_msgs roscpp roslib rospy sensor_msgs std_msgs tf tf2
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  # include
  ${catkin_INCLUDE_DIRS}
)
  
catkin_python_setup()


############################################################
# CMakelist for optimal_control node
############################################################

# Install Python scripts
catkin_install_python(PROGRAMS
  src/amr_control/optimal_control.py 
DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

############################################################
# CMakelist for casadi_example node
############################################################

# Install Python scripts
catkin_install_python(PROGRAMS
  src/amr_control/tutorials/single_shooting_nmpc_tutorial.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

############################################################
# CMakelist for casadi_example node
############################################################

# Install Python scripts
catkin_install_python(PROGRAMS
  src/amr_control/tutorials/multiple_shooting_nmpc_tutorial.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

############################################################
# CMakelist for multiple_shooting_nmpc_obstacle_avoidance_tutorial node
############################################################

# Install Python scripts
catkin_install_python(PROGRAMS
  src/amr_control/tutorials/multiple_shooting_nmpc_obstacle_avoidance_tutorial.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

############################################################
# CMakelist for multiple_shooting_nmpc_obstacle_avoidance_refrence_tracking node
############################################################

# Install Python scripts
catkin_install_python(PROGRAMS
  src/amr_control/tutorials/multiple_shooting_nmpc_obstacle_avoidance_refrence_tracking.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
