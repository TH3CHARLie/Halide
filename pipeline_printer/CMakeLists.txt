cmake_minimum_required(VERSION 3.28)
project(pipeline_printer)

enable_testing()

# Set up language settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

# Find Halide
find_package(Halide REQUIRED)

add_executable(pipeline_printer main.cpp)
target_link_directories(pipeline_printer PRIVATE /home/xuanday/dev/Halide/install/lib)
target_link_libraries(pipeline_printer
                      PRIVATE
                      Halide
                      Halide::ImageIO
                      Halide::Tools)