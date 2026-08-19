cmake_minimum_required(VERSION 3.16)
project(phywam_plugins CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-O2 -Wall -Wno-unused-parameter)
# Drop the transitive libav/libpulse/libsystemd chain we never call: conda ships
# a libsystemd newer than the host glibc and dlopen of the plugin fails on it.
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--as-needed")

find_package(gz-sim8 REQUIRED)
find_package(gz-plugin2 REQUIRED COMPONENTS register)
find_package(gz-transport13 REQUIRED)
find_package(gz-msgs10 REQUIRED)
find_package(gz-math7 REQUIRED)

add_library(PhyWamPlant SHARED PhyWamPlant.cc)
add_library(PhyWamTraffic SHARED PhyWamTraffic.cc)
target_link_libraries(PhyWamPlant PRIVATE
  gz-sim8::gz-sim8
  gz-plugin2::register
  gz-transport13::gz-transport13
  gz-msgs10::gz-msgs10
  gz-math7::gz-math7)
target_link_libraries(PhyWamTraffic PRIVATE
  gz-sim8::gz-sim8
  gz-plugin2::register
  gz-transport13::gz-transport13
  gz-msgs10::gz-msgs10
  gz-math7::gz-math7)
