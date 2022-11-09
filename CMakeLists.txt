cmake_minimum_required(VERSION 3.15)
project(main LANGUAGES CXX CUDA)
set(CMAKE_BUILD_TYPE Release) 
# set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS_DEBUG "-ggdb3 -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g -freciprocal-math -mtune=native -march=native")

set(CMAKE_CUDA_FLAGS_DEBUG "-G -g")
set(CMAKE_CUDA_FLAGS_RELEASE " -lineinfo")

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)
FetchContent_Declare(fmt        GIT_PROGRESS ON GIT_REPOSITORY https://github.com/fmtlib/fmt.git)
FetchContent_Declare(glm        GIT_PROGRESS ON GIT_REPOSITORY https://github.com/g-truc/glm.git)
FetchContent_MakeAvailable(fmt glm)

add_executable(
    main 
    main.cu
)
target_link_libraries(main PUBLIC fmt glm)
