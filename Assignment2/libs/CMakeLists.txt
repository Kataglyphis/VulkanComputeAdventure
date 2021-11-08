add_subdirectory(glm)

set(GLFW_BUILD_EXAMPLES OFF CACHE INTERNAL "Build the GLFW example programs")
set(GLFW_BUILD_TESTS OFF CACHE INTERNAL "Build the GLFW test programs")
set(GLFW_BUILD_DOCS OFF CACHE INTERNAL "Build the GLFW documentation")
set(GLFW_INSTALL OFF CACHE INTERNAL "Generate installation target")
#add_subdirectory(glfw)

### allows you to just put stb in the "target link library"
add_library(csv INTERFACE)
target_include_directories(csv INTERFACE CSVWriter/include)

add_library(stb INTERFACE)
target_include_directories(stb INTERFACE stb/)

#add_library(vma INTERFACE)
#target_include_directories(vma INTERFACE VulkanMemoryAllocator/include)