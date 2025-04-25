# Commands for GMapping
## Simulation starten
roslaunch sensor_fusion run_simulation.launch model:=burger

## GMapping ausführen mit dem Launch File im launch Ordner
roslaunch sensor_fusion gmapping.launch model:=burger

## Mit dem Turtlebot herumfahren und Area scannen
rosrun turtlebot3_teleop turtlebot3_teleop_key

## Map speichern und evt. mit Gimp bearbeiten
rosrun map_server map_saver -f /home/kevin/RobotikMaster/Semester2/Probabilistic_Lab/src/sensor_fusion/maps/small_house