cmake_minimum_required(VERSION 3.20)

project(consumer_app
    VERSION 1.0.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find the installed SSMForge package configuration
find_package(SSMForge CONFIG REQUIRED)

# Add consumer executable
add_executable(consumer_app
    main.cpp
)

# Link against the installed SSMForge library via its namespaced target
target_link_libraries(consumer_app
    PRIVATE
        SSMForge::ssmforge
)

