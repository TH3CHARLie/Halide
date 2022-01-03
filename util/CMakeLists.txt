add_executable(HalideTraceViz HalideTraceViz.cpp)
target_link_libraries(HalideTraceViz PRIVATE Halide::Halide Halide::Tools)

add_executable(HalideTraceDump HalideTraceDump.cpp HalideTraceUtils.cpp)
target_link_libraries(HalideTraceDump PRIVATE Halide::Halide Halide::ImageIO Halide::Tools)

add_executable(HalideTraceCounter HalideTraceCounter.cpp)
target_link_libraries(HalideTraceCounter PRIVATE Halide::Halide Halide::Tools)
