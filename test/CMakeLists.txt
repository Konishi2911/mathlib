add_executable(cubic_spline_test interpolation/cubic_spline.cc)
target_link_libraries(cubic_spline_test PRIVATE ${BLAS_LIBRARIES} ${OpenMP_CXX_LIBRARIES} GTest::GTest GTest::Main)
gtest_discover_tests(cubic_spline_test)

add_executable(secant_test roots/secant.cc)
target_link_libraries(secant_test PRIVATE ${BLAS_LIBRARIES} ${OpenMP_CXX_LIBRARIES} GTest::GTest GTest::Main)
gtest_discover_tests(secant_test)

add_executable(simpson_test integral/simpson.cc)
target_link_libraries(simpson_test PRIVATE ${BLAS_LIBRARIES} ${OpenMP_CXX_LIBRARIES} GTest::GTest GTest::Main)
gtest_discover_tests(simpson_test)

add_executable(steepest_descent_test nlp/steepest_descent.cc)
target_link_libraries(steepest_descent_test PRIVATE ${BLAS_LIBRARIES} ${OpenMP_CXX_LIBRARIES} GTest::GTest GTest::Main)
gtest_discover_tests(steepest_descent_test)

add_executable(nelder_mead_test nlp/nelder_mead.cc)
target_link_libraries(nelder_mead_test PRIVATE ${BLAS_LIBRARIES} ${OpenMP_CXX_LIBRARIES} GTest::GTest GTest::Main)
gtest_discover_tests(nelder_mead_test)