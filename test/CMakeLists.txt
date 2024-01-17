add_executable(cubic_spline_test interpolation/cubic_spline.cc)
target_link_libraries(cubic_spline_test PRIVATE ${BLAS_LIBRARIES} ${OpenMP_CXX_LIBRARIES} GTest::GTest GTest::Main)
gtest_discover_tests(cubic_spline_test)