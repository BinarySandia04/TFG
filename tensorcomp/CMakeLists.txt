cmake_minimum_required(VERSION 3.16)
project(TensorComp VERSION 0.1)

SET(CMAKE_COLOR_MAKEFILE ON)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(PROJECT_NAME TensorComp)
set(CMAKE_INSTALL_MESSAGE LAZY)

file(GLOB src "src/*.h" "src/*.cpp")
source_group("src" FILES ${src})

set(ALL_FILES ${src})
set(ROOT_NAMESPACE TensorComp)

add_executable(${PROJECT_NAME} ${ALL_FILES})

# target_include_directories(${PROJECT_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../external") 
# target_include_directories(${PROJECT_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../external/argh/include")
# target_include_directories(${PROJECT_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../external/termcolor/include")
# target_include_directories(${PROJECT_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../external/tomlplusplus/include")

set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME "tensorcomp")
install(TARGETS ${PROJECT_NAME} DESTINATION bin)