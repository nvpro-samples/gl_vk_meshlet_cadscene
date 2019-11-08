cmake_minimum_required(VERSION 2.8)

get_filename_component(PROJNAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
Project(${PROJNAME})
Message(STATUS "-------------------------------")
Message(STATUS "Processing Project ${PROJNAME}:")


SET(MODEL_DOWNLOAD_${PROJNAME} ON CACHE BOOL "CHECK to download the 3d models needed for this sample" )

#####################################################################################
# look for shared_sources 1) as a sub-folder 2) at some other locations
# this cannot be put anywhere else since we still didn't find CMakeLists_include.txt yet
#
if(NOT BASE_DIRECTORY) # if not defined, it means this cmake file was called as the first entry point and not included
  # check if the external repository is outside or inside the project (as a sub-module)
  # testing the file CMakeLists_include.txt because when sub-modules are not cloned, the folders are still there...
  # we also assume here that if shared_sources is there, shared_external is, too...
  SET(ADD_SUBDIR_BELOW 1)
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/shared_sources/CMakeLists_include.txt)
    Message(STATUS "found sub-folder shared_sources")
    SET(BASE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    INCLUDE(${CMAKE_CURRENT_SOURCE_DIR}/shared_sources/CMakeLists_include.txt)
  elseif(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../../shared_sources)
    SET(BASE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../..)
    INCLUDE(${CMAKE_CURRENT_SOURCE_DIR}/../../shared_sources/CMakeLists_include.txt)
  else()
    SET(BASE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..)
    INCLUDE(${CMAKE_CURRENT_SOURCE_DIR}/../shared_sources/CMakeLists_include.txt)
  endif()
endif(NOT BASE_DIRECTORY)
_add_project_definitions(${PROJNAME})

set( BUILD_${PROJNAME}_VULKAN_ONLY FALSE CACHE BOOL "Avoids OpenGL in samples that support dual use" )

#####################################################################################
# additions from packages needed for this sample
# add refs  in LIBRARIES_OPTIMIZED
# add refs  in LIBRARIES_DEBUG
# add files in PACKAGE_SOURCE_FILES

set( VK_EXENAME "vk_meshlet_cadscene" )
set( GL_VK_EXENAME "gl_vk_meshlet_cadscene" )

if(NOT BUILD_${PROJNAME}_VULKAN_ONLY)
  _add_package_OpenGL()
endif()
_add_package_VulkanSDK()
_add_package_ShaderC()
_add_package_IMGUI()
_add_package_ZLIB()

#####################################################################################
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()


#####################################################################################
# process the rest of some cmake code that needs to be done *after* the packages add
_add_shared_sources_lib()

add_definitions(-DCSF_ZIP_SUPPORT=1)

#####################################################################################
# download models
unset(ASSET_FILES)
if(MODEL_DOWNLOAD_${PROJNAME})
  set(FILELIST 
  "worldcar.csf.gz"
  "blade.csf.gz"
  )
  _download_files(
    "${FILELIST}"
    "http://developer.download.nvidia.com/ProGraphics/nvpro-samples"
    ${DOWNLOAD_TARGET_DIR}
    0
    ASSET_FILES
  )
endif()

#####################################################################################
# Source files for this project
#
file(GLOB GL_VK_SOURCE_FILES *.cpp *.hpp *.inl *.h *.c)
file(GLOB VK_SOURCE_FILES *.cpp *.hpp *.inl *.h *.c)
file(GLOB GLSL_FILES *.glsl)

list(REMOVE_ITEM VK_SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/cadscene_gl.cpp)
list(REMOVE_ITEM VK_SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/cadscene_gl.hpp)
list(REMOVE_ITEM VK_SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/gl_nv_mesh_shader.h)
list(REMOVE_ITEM VK_SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/gl_nv_mesh_shader.cpp)
list(REMOVE_ITEM VK_SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/gl_nv_draw_vulkan_image.h)
list(REMOVE_ITEM VK_SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/gl_nv_draw_vulkan_image.cpp)
list(REMOVE_ITEM VK_SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/resources_gl.cpp)
list(REMOVE_ITEM VK_SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/resources_gl.hpp)
list(REMOVE_ITEM VK_SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/renderer_gl.cpp)
list(REMOVE_ITEM VK_SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/renderer_gl_mesh.cpp)



#####################################################################################
# common source code needed for this sample
#
source_group(common FILES 
  ${COMMON_SOURCE_FILES}
  ${PACKAGE_SOURCE_FILES}
)
source_group(shaders FILES 
  ${GLSL_FILES}
)

source_group("Source Files" FILES ${GL_VK_SOURCE_FILES})

if(UNIX)
  set(UNIXLINKLIBS dl pthread)
else()
  set(UNIXLINKLIBS)
endif()


#####################################################################################
# Executable
#
if(WIN32 AND NOT GLUT_FOUND)
  add_definitions(/wd4996) #remove printf warning
  add_definitions(/wd4244) #remove double to float conversion warning
  add_definitions(/wd4305) #remove double to float truncation warning
else()
  add_definitions(-fpermissive)
endif()


add_executable(${VK_EXENAME} ${VK_SOURCE_FILES} ${COMMON_SOURCE_FILES} ${PACKAGE_SOURCE_FILES} ${GLSL_FILES})

if(NOT BUILD_${PROJNAME}_VULKAN_ONLY)
  add_executable(${GL_VK_EXENAME} ${GL_VK_SOURCE_FILES} ${COMMON_SOURCE_FILES} ${PACKAGE_SOURCE_FILES} ${GLSL_FILES})
  target_compile_definitions(${GL_VK_EXENAME} PRIVATE -DHAS_OPENGL=1)
endif()

#####################################################################################
# Linkage
#

target_link_libraries(${VK_EXENAME} ${PLATFORM_LIBRARIES} shared_sources)

foreach(DEBUGLIB ${LIBRARIES_DEBUG})
  target_link_libraries(${VK_EXENAME} debug ${DEBUGLIB})
endforeach(DEBUGLIB)

foreach(RELEASELIB ${LIBRARIES_OPTIMIZED})
  target_link_libraries(${VK_EXENAME} optimized ${RELEASELIB})
endforeach(RELEASELIB)

if(NOT BUILD_${PROJNAME}_VULKAN_ONLY)
  target_link_libraries(${GL_VK_EXENAME} ${PLATFORM_LIBRARIES} shared_sources)

  foreach(DEBUGLIB ${LIBRARIES_DEBUG})
    target_link_libraries(${GL_VK_EXENAME} debug ${DEBUGLIB})
  endforeach(DEBUGLIB)

  foreach(RELEASELIB ${LIBRARIES_OPTIMIZED})
    target_link_libraries(${GL_VK_EXENAME} optimized ${RELEASELIB})
  endforeach(RELEASELIB)
endif()


#####################################################################################
# copies binaries that need to be put next to the exe files (ZLib, etc.)
#

_copy_binaries_to_target( ${VK_EXENAME} )
if(NOT BUILD_${PROJNAME}_VULKAN_ONLY)
  _copy_binaries_to_target( ${GL_VK_EXENAME} )
endif()

LIST(APPEND GLSL_FILES "common.h")
LIST(APPEND GLSL_FILES "config.h")
LIST(APPEND ASSET_FILES "blade_meshlet.cfg")
LIST(APPEND ASSET_FILES "blade_meshlet_viewpoints.txt")
LIST(APPEND ASSET_FILES "worldcar_meshlet.cfg")
LIST(APPEND ASSET_FILES "worldcar_meshlet_viewpoints.txt")

install(FILES ${GLSL_FILES} CONFIGURATIONS Release DESTINATION "bin_${ARCH}/GLSL_${PROJNAME}")
install(FILES ${GLSL_FILES} CONFIGURATIONS Debug DESTINATION "bin_${ARCH}_debug/GLSL_${PROJNAME}")
install(FILES ${ASSET_FILES} CONFIGURATIONS Debug DESTINATION "bin_${ARCH}_debug")
install(FILES ${ASSET_FILES} CONFIGURATIONS Release DESTINATION "bin_${ARCH}")
