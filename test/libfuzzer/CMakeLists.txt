tests(GROUPS libfuzzer
      SOURCES
      fuzzer_valid.cpp
      )

set(LIB_FUZZING_ENGINE "$ENV{LIB_FUZZING_ENGINE}"
    CACHE STRING "Compiler flags necessary to link the fuzzing engine of choice e.g. libfuzzer, afl etc.")

foreach(fuzzer "libfuzzer_fuzzer_valid")
  target_link_libraries(${fuzzer} PRIVATE Halide::Halide)

  # Allow OSS-fuzz to manage flags directly
  if (LIB_FUZZING_ENGINE)
    target_link_libraries(${fuzzer} PRIVATE "${LIB_FUZZING_ENGINE}")
  else ()
    # By default just build with address-sanitizers/libfuzzer for local testing
    target_compile_options(${fuzzer} PRIVATE -fsanitize=fuzzer-no-link)
    target_link_options(${fuzzer} PRIVATE -fsanitize=fuzzer)
  endif ()
endforeach()
