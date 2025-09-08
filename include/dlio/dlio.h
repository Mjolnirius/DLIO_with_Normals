/***********************************************************
 *                                                         *
 * Copyright (c)                                           *
 *                                                         *
 * The Verifiable & Control-Theoretic Robotics (VECTR) Lab *
 * University of California, Los Angeles                   *
 *                                                         *
 * Authors: Kenny J. Chen, Ryan Nemiroff, Brett T. Lopez   *
 * Contact: {kennyjchen, ryguyn, btlopez}@ucla.edu         *
 *                                                         *
 ***********************************************************/

// SYSTEM
#include <atomic>

#ifdef HAS_CPUID
#include <cpuid.h>
#endif

#include <ctime>
#include <fstream>
#include <future>
#include <iomanip>
#include <ios>
#include <iostream>
#include <mutex>
#include <signal.h>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sys/times.h>
#include <thread>

template <typename T>
std::string to_string_with_precision(const T a_value, const int n = 6)
{
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << a_value;
    return out.str();
}

// BOOST
#include <boost/format.hpp>

// PCL
#define PCL_NO_PRECOMPILE

// DLIO
#include <nano_gicp/nano_gicp.h>

// NEU: für Normalen-Makro
#include <pcl/point_types.h>

namespace dlio {
  enum class SensorType { OUSTER, VELODYNE, HESAI, LIVOX, UNKNOWN };

  class OdomNode;
  class MapNode;

  struct Point {
    Point() : data{0.f, 0.f, 0.f, 1.f}, intensity(0.f), curvature(0.f) {}

    PCL_ADD_POINT4D;      // x,y,z, w (Padding)
    PCL_ADD_NORMAL4D;     // normal_x, normal_y, normal_z, normal_w (Padding)

    float curvature;      // added
    float intensity;

    union {
      std::uint32_t t;            // (Ouster) ns seit Scan-Beginn
      float         time;         // (Velodyne) s seit Scan-Beginn
      double        timestamp;    // (Hesai/Livox) absolute Zeit
    };

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  } EIGEN_ALIGN16;
}


POINT_CLOUD_REGISTER_POINT_STRUCT(dlio::Point,
  (float, x, x)
  (float, y, y)
  (float, z, z)
  (float, normal_x, normal_x)   // added
  (float, normal_y, normal_y)   // added
  (float, normal_z, normal_z)   // added
  (float, curvature, curvature) // added
  (float, intensity, intensity) 
  (std::uint32_t, t, t)
  (float, time, time)
  (double, timestamp, timestamp)
)

typedef dlio::Point PointType;

