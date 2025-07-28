# Install script for directory: /storage1/fs1/yajiey/Active/Thomas/Aperture4/problems

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/storage1/fs1/yajiey/Active/Thomas/Aperture4/problems/examples/cmake_install.cmake")
  include("/storage1/fs1/yajiey/Active/Thomas/Aperture4/problems/sph_wave_test/cmake_install.cmake")
  include("/storage1/fs1/yajiey/Active/Thomas/Aperture4/problems/debug/cmake_install.cmake")
  include("/storage1/fs1/yajiey/Active/Thomas/Aperture4/problems/two_stream/cmake_install.cmake")
  include("/storage1/fs1/yajiey/Active/Thomas/Aperture4/problems/reconnection/cmake_install.cmake")
  include("/storage1/fs1/yajiey/Active/Thomas/Aperture4/problems/gr_2d_kerr_schild/cmake_install.cmake")
  include("/storage1/fs1/yajiey/Active/Thomas/Aperture4/problems/magnetar/cmake_install.cmake")
  include("/storage1/fs1/yajiey/Active/Thomas/Aperture4/problems/parker/cmake_install.cmake")
  include("/storage1/fs1/yajiey/Active/Thomas/Aperture4/problems/alfven_wave_2d/cmake_install.cmake")
  include("/storage1/fs1/yajiey/Active/Thomas/Aperture4/problems/vlasov_test/cmake_install.cmake")
  include("/storage1/fs1/yajiey/Active/Thomas/Aperture4/problems/fast_wave_2d/cmake_install.cmake")
  include("/storage1/fs1/yajiey/Active/Thomas/Aperture4/problems/fast_wave_2d_cartesian/cmake_install.cmake")
  include("/storage1/fs1/yajiey/Active/Thomas/Aperture4/problems/fast_wave_3d_cartesian/cmake_install.cmake")

endif()

