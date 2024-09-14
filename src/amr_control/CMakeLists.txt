cmake_minimum_required(VERSION 2.8.3)
project(amr_control)

## Füge Abhängigkeiten hinzu
find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  sensor_msgs
  nav_msgs
  tf
  dynamic_reconfigure
)

## Füge die generierten dynamischen Rekonfigurationsoptionen hinzu
# ACHTUNG: Diese Zeile muss vor dem catkin_package-Aufruf kommen!
generate_dynamic_reconfigure_options(
  cfg/ObstacleDetectionConfig.cfg
)

## catkin spezifische Einstellungen
catkin_package(
  CATKIN_DEPENDS dynamic_reconfigure
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)


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

############################################################
# CMakelist for obstacle_detection node
############################################################

# Install Python scripts
catkin_install_python(PROGRAMS
  src/amr_control/obstacle_detection.py 
DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

############################################################
# CMakelist for trajectory_interpolation_test node
############################################################

# Install Python scripts
catkin_install_python(PROGRAMS
  src/amr_control/trajectory_interpolation_test.py 
DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

############################################################
# CMakelist for move_base_execution node
############################################################

# Install Python scripts
catkin_install_python(PROGRAMS
  scripts/move_base_execution.py 
DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

############################################################
# CMakelist for path_recorder node
############################################################

# Install Python scripts
catkin_install_python(PROGRAMS
  scripts/path_recorder.py 
DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

############################################################
# CMakelist for move_urdf_box node
############################################################

# Install Python scripts
catkin_install_python(PROGRAMS
  scripts/move_urdf_box.py 
DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

############################################################
# CMakelist for plot_paths node
############################################################

# Install Python scripts
catkin_install_python(PROGRAMS
  scripts/plot_paths.py 
DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
