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

catkin_python_setup()

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
# Installiere Python-Module
############################################################

# Kopiere das gesamte amr_control Python-Paket in den Python-Installationspfad
install(
  DIRECTORY src/amr_control/
  DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}
  FILES_MATCHING PATTERN "*.py"
)

############################################################
# Installiere Python-Skripte (Executable Nodes)
############################################################

catkin_install_python(PROGRAMS
  src/amr_control/optimal_control.py
  src/amr_control/obstacle_detection.py
  src/amr_control/trajectory_interpolation_test.py
  scripts/move_base_execution.py
  scripts/path_recorder.py
  scripts/move_urdf_box.py
  scripts/plot_paths.py
DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

############################################################
# Installiere Tutorials
############################################################

catkin_install_python(PROGRAMS
  src/amr_control/tutorials/single_shooting_nmpc_tutorial.py
  src/amr_control/tutorials/multiple_shooting_nmpc_tutorial.py
  src/amr_control/tutorials/multiple_shooting_nmpc_obstacle_avoidance_tutorial.py
  src/amr_control/tutorials/multiple_shooting_nmpc_obstacle_avoidance_refrence_tracking.py
DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
