add_executable(mathlib_cubic_spline_test interpolation/cubic_spline.cc)
target_link_libraries(mathlib_cubic_spline_test PRIVATE ${BLAS_LIBRARIES} ${OpenMP_CXX_LIBRARIES} GTest::GTest GTest::Main)
gtest_discover_tests(mathlib_cubic_spline_test)

add_executable(mathlib_linear_intrpl_test interpolation/linear.cc)
target_link_libraries(mathlib_linear_intrpl_test PRIVATE ${BLAS_LIBRARIES} ${OpenMP_CXX_LIBRARIES} GTest::GTest GTest::Main)
gtest_discover_tests(mathlib_linear_intrpl_test)

add_executable(mathlib_secant_test roots/secant.cc)
target_link_libraries(mathlib_secant_test PRIVATE ${BLAS_LIBRARIES} ${OpenMP_CXX_LIBRARIES} GTest::GTest GTest::Main)
gtest_discover_tests(mathlib_secant_test)

add_executable(mathlib_simpson_test integral/simpson.cc)
target_link_libraries(mathlib_simpson_test PRIVATE ${BLAS_LIBRARIES} ${OpenMP_CXX_LIBRARIES} GTest::GTest GTest::Main)
gtest_discover_tests(mathlib_simpson_test)

add_executable(mathlib_cost_function_test nlp/cost_function.cc)
target_link_libraries(mathlib_cost_function_test PRIVATE ${BLAS_LIBRARIES} ${OpenMP_CXX_LIBRARIES} GTest::GTest GTest::Main)
gtest_discover_tests(mathlib_cost_function_test)

add_executable(mathlib_steepest_descent_test nlp/steepest_descent.cc)
target_link_libraries(mathlib_steepest_descent_test PRIVATE ${BLAS_LIBRARIES} ${OpenMP_CXX_LIBRARIES} GTest::GTest GTest::Main)
gtest_discover_tests(mathlib_steepest_descent_test)

add_executable(mathlib_lm_test nlp/levenberg_marquardt.cc)
target_link_libraries(mathlib_lm_test PRIVATE ${BLAS_LIBRARIES} ${OpenMP_CXX_LIBRARIES} GTest::GTest GTest::Main)
gtest_discover_tests(mathlib_lm_test)

add_executable(mathlib_nelder_mead_test nlp/nelder_mead.cc)
target_link_libraries(mathlib_nelder_mead_test PRIVATE ${BLAS_LIBRARIES} ${OpenMP_CXX_LIBRARIES} GTest::GTest GTest::Main)
gtest_discover_tests(mathlib_nelder_mead_test)

add_executable(mathlib_polynomial_test polynomial.cc)
target_link_libraries(mathlib_polynomial_test PRIVATE ${BLAS_LIBRARIES} ${OpenMP_CXX_LIBRARIES} GTest::GTest GTest::Main)
gtest_discover_tests(mathlib_polynomial_test)