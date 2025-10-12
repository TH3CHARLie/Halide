tests(GROUPS libfuzzer
      SOURCES
      fuzzer_valid.cpp
      )


foreach(fuzzer "libfuzzer_fuzzer_valid")
    target_link_libraries(${fuzzer} PRIVATE Halide::Halide)

    # By default just build with address-sanitizers/libfuzzer for local testing
    target_compile_options(${fuzzer} PRIVATE -fsanitize=fuzzer-no-link)
    target_link_options(${fuzzer} PRIVATE -fsanitize=fuzzer)
endforeach()