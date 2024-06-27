cmake_minimum_required(VERSION 3.0.2)
project(fyp)


find_package(catkin REQUIRED
    std_msgs # <-- Add these
    message_generation)

add_service_files(
  FILES
  RobotServer.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs  # Or other packages containing msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES fyp
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
)



###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
# ${catkin_INCLUDE_DIRS}
)

