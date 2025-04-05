cmake_minimum_required(VERSION 3.14)

project(gpu-math)

file(GLOB_RECURSE PROJECT_SOURCES "sources/*.cxx")
file(GLOB_RECURSE PROJECT_INCLUDE "include/*.hxx")

add_executable(${PROJECT_NAME} ${PROJECT_SOURCES})
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_INCLUDE})
