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

/***********************************************************************
 * BSD 3-Clause License
 *
 * Copyright (c) 2020, SMRT-AIST
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *************************************************************************/

#include "dlio/dlio.h"
#include "nano_gicp/nano_gicp.h"

#include <typeinfo>
#include <pcl/point_traits.h>   // pcl::traits::has_field

#include <execinfo.h>
static inline void print_bt() {
  void* buf[64];
  int n = backtrace(buf, 64);
  char** syms = backtrace_symbols(buf, n);
  for (int i = 0; i < n; ++i) std::cerr << syms[i] << "\n";
  free(syms);
}


template class nano_gicp::NanoGICP<PointType, PointType>;

namespace nano_gicp {

template <typename PointSource, typename PointTarget>
NanoGICP<PointSource, PointTarget>::NanoGICP() {
#ifdef _OPENMP
  num_threads_ = omp_get_max_threads();
#else
  num_threads_ = 1;
#endif

  k_correspondences_ = 20;
  reg_name_ = "NanoGICP";
  corr_dist_threshold_ = std::numeric_limits<float>::max();

  regularization_method_ = RegularizationMethod::PLANE;
}

template <typename PointSource, typename PointTarget>
NanoGICP<PointSource, PointTarget>::~NanoGICP() {}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setNumThreads(int n) {
  num_threads_ = n;

#ifdef _OPENMP
  if (n == 0) {
    num_threads_ = omp_get_max_threads();
  }
#endif
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setCorrespondenceRandomness(int k) {
  k_correspondences_ = k;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setMaxCorrespondenceDistance(double corr) {
  corr_dist_threshold_ = corr;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setRegularizationMethod(RegularizationMethod method) {
  regularization_method_ = method;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::swapSourceAndTarget() {
  input_.swap(target_);
  source_kdtree_.swap(target_kdtree_);
  source_covs_.swap(target_covs_);

  correspondences_.clear();
  sq_distances_.clear();
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::clearSource() {
  input_.reset();
  source_covs_.reset();
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::clearTarget() {
  target_.reset();
  target_covs_.reset();
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::registerInputSource(const PointCloudSourceConstPtr& cloud) {
  if (input_ == cloud) {
    return;
  }
  pcl::Registration<PointSource, PointTarget, Scalar>::setInputSource(cloud);
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::registerInputTarget(const PointCloudTargetConstPtr& cloud) {
  if (target_ == cloud) {
    return;
  }
  pcl::Registration<PointSource, PointTarget, Scalar>::setInputTarget(cloud);
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setInputSource(const PointCloudSourceConstPtr& cloud) {
  if (input_ == cloud) {
    return;
  }

  pcl::Registration<PointSource, PointTarget, Scalar>::setInputSource(cloud);

  std::shared_ptr<nanoflann::KdTreeFLANN<PointSource>> source_kdtree = std::make_shared<nanoflann::KdTreeFLANN<PointSource>>();
  source_kdtree->setInputCloud(cloud);
  source_kdtree_ = source_kdtree;

  source_covs_.reset();
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setInputTarget(const PointCloudTargetConstPtr& cloud) {
  if (target_ == cloud) {
    return;
  }
  pcl::Registration<PointSource, PointTarget, Scalar>::setInputTarget(cloud);

  std::shared_ptr<nanoflann::KdTreeFLANN<PointTarget>> target_kdtree = std::make_shared<nanoflann::KdTreeFLANN<PointTarget>>();
  target_kdtree->setInputCloud(cloud);
  target_kdtree_ = target_kdtree;

  target_covs_.reset();
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setSourceCovariances(const std::shared_ptr<const CovarianceList>& covs) {
  source_covs_ = covs;
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::setTargetCovariances(const std::shared_ptr<const CovarianceList>& covs) {
  target_covs_ = covs;
}

// ----------------------------------------------------------------------------------------------------------------------------------------------
// ------------------------------------------------------------
// retrieveSourceCovariancesFromROSMsg
// ------------------------------------------------------------
template <typename PointSource, typename PointTarget>
bool NanoGICP<PointSource, PointTarget>::retrieveSourceCovariancesFromROSMsg() {
  std::cout << "[NanoGICP] retrieveSourceCovariancesFromROSMsg() called!" << std::endl;

  if (!input_) {
    std::cerr << "[NanoGICP] input_ is null\n";
    return false;
  }

  
  bool debug_normals_ = true;
  // DEBUGGING
  if (debug_normals_) {
    debugDumpSourceCloudOnce_();
  }

  using T = PointSource;
  constexpr bool has_normals =
      pcl::traits::has_field<T, pcl::fields::normal_x>::value &&
      pcl::traits::has_field<T, pcl::fields::normal_y>::value &&
      pcl::traits::has_field<T, pcl::fields::normal_z>::value;

  bool force_standard_covariance_calculation_ = false; // for testing

  if (has_normals && !force_standard_covariance_calculation_) {
    return calculateCovariancesFromLiSuNormals();
  } else {
    // Fallback: standard GICP covariance calculation
    std::cerr << "[NanoGICP] No normals in PointType / or Standard way was forced → using calculateSourceCovariances()" << std::endl;
    return calculateSourceCovariances();
  }
}



// ------------------------------------------------------------
// calculateCovariancesFromLiSuNormals
// ------------------------------------------------------------
template <typename PointSource, typename PointTarget>
bool NanoGICP<PointSource, PointTarget>::calculateCovariancesFromLiSuNormals() {
  std::cout << "[NanoGICP] calculateCovariancesFromLiSuNormals() called!" << std::endl;

  if (!input_ || input_->empty()) {
    std::cerr << "[NanoGICP] input_ empty in calculateCovariancesFromLiSuNormals\n";
    return false;
  }

  using T = PointSource;


  // ---- Parameter (update later - tuning) ----
  const double s   = 0.05;    // charakteristischer Längenscale [m], z.B. Voxelgröße
  const double r   = 0.15;    // Anisotropie: sigma_n / sigma_t  (normal schmaler)
  const double st2 = 1;       //s * s;
  const double sn2 = 1e3;     //(r * s) * (r * s);
  const double eps = 1e-8;    //  Regularisierung für Invertierbarkeit

  const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();

  auto covs = std::make_shared<CovarianceList>();
  covs->resize(input_->size(), Eigen::Matrix4d::Zero());

  for (std::size_t i = 0; i < input_->size(); ++i) {
    const auto& p = input_->points[i];

    // Normale holen & (re-)normalisieren (VoxelGrid mittelt Normalen nur arithmetisch) -- 
    // VoxelGrid update (normalen anstelle mitteln, einen repr. punkt auswählen) - noch im Backlog
    Eigen::Vector3d n(static_cast<double>(p.normal_x),
                      static_cast<double>(p.normal_y),
                      static_cast<double>(p.normal_z));

    if (!std::isfinite(n.x()) || !std::isfinite(n.y()) || !std::isfinite(n.z())) {
      n = Eigen::Vector3d(0, 0, 1);
    }
    double nrm = n.norm();
    if (nrm > 1e-12) n /= nrm; else n = Eigen::Vector3d(0, 0, 1);

    // Projektoren entlang / orthogonal zur Normalen
    const Eigen::Matrix3d Ppar  = n * n.transpose();   // n n^T
    const Eigen::Matrix3d Pperp = I - Ppar;            // I - n n^T

    // Σ = σ_t^2 P_perp + σ_n^2 P_par  (+ eps*I)
    Eigen::Matrix3d Sigma = st2 * Pperp + sn2 * Ppar;
    Sigma.diagonal().array() += eps;

    // 3×3 → 4×4 einbetten
    Eigen::Matrix4d Q = Eigen::Matrix4d::Zero();
    Q.block<3,3>(0,0) = Sigma;
    Q(3,3) = 1.0; // in update_correspondences() wird (3,3) vor der Inversion auf 1 gesetzt und danach auf 0

    (*covs)[i] = Q;
  }

  source_covs_    = covs;
  source_density_ = 0.f; // optional

  return true;
}





// ------------------------------------------------------------
// Debug Dump Source Cloud Once
// ------------------------------------------------------------
template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::debugDumpSourceCloudOnce_() {
  using T = PointSource;

  std::cerr << "  PointSource type: " << typeid(T).name() << "\n";
  std::cerr << "  cloud size: " << input_->size() << "\n";

  const bool has_x   = pcl::traits::has_field<T, pcl::fields::x>::value;
  const bool has_y   = pcl::traits::has_field<T, pcl::fields::y>::value;
  const bool has_z   = pcl::traits::has_field<T, pcl::fields::z>::value;
  const bool has_i   = pcl::traits::has_field<T, pcl::fields::intensity>::value;
  const bool has_nx  = pcl::traits::has_field<T, pcl::fields::normal_x>::value;
  const bool has_ny  = pcl::traits::has_field<T, pcl::fields::normal_y>::value;
  const bool has_nz  = pcl::traits::has_field<T, pcl::fields::normal_z>::value;
  const bool has_curv= pcl::traits::has_field<T, pcl::fields::curvature>::value;

  std::cerr << "  has x,y,z    : " << has_x << "," << has_y << "," << has_z << "\n";
  std::cerr << "  has intensity: " << has_i << "\n";
  std::cerr << "  has normals  : "
            << (has_nx && has_ny && has_nz ? "yes" : "no")
            << " (nx=" << has_nx << ", ny=" << has_ny << ", nz=" << has_nz << ")\n";
  std::cerr << "  has curvature: " << has_curv << "\n";

  if (input_->empty()) {
    std::cerr << "  cloud is empty\n";
    return;
  }

  const auto& p0 = input_->points.front();
  std::cerr << "  --- First point (raw) ---\n";
  if constexpr (pcl::traits::has_field<T, pcl::fields::x>::value) {
    std::cerr << "    p0.xyz = [" << p0.x << ", " << p0.y << ", " << p0.z << "]\n";
  }
  if constexpr (pcl::traits::has_field<T, pcl::fields::intensity>::value) {
    std::cerr << "    p0.intensity = " << p0.intensity << "\n";
  }
  if constexpr (pcl::traits::has_field<T, pcl::fields::normal_x>::value &&
                pcl::traits::has_field<T, pcl::fields::normal_y>::value &&
                pcl::traits::has_field<T, pcl::fields::normal_z>::value) {
    std::cerr << "    p0.normal = [" << p0.normal_x << ", "
              << p0.normal_y << ", " << p0.normal_z << "]\n";
  }
  if constexpr (pcl::traits::has_field<T, pcl::fields::curvature>::value) {
    std::cerr << "    p0.curvature = " << p0.curvature << "\n";
  }

  /* // show first non-zero point - it's the same as p0 
  for (std::size_t i = 0; i < input_->size(); ++i) {
    const auto& pt = input_->points[i];
    const double norm = std::sqrt(pt.x*pt.x + pt.y*pt.y + pt.z*pt.z);
    if (norm > 1e-6) {
      std::cerr << "  --- First non-zero point at index " << i << " ---\n";
      std::cerr << "    xyz = [" << pt.x << ", " << pt.y << ", " << pt.z << "]\n";
      if constexpr (pcl::traits::has_field<T, pcl::fields::intensity>::value) {
        std::cerr << "    intensity = " << pt.intensity << "\n";
      }
      if constexpr (pcl::traits::has_field<T, pcl::fields::normal_x>::value &&
                    pcl::traits::has_field<T, pcl::fields::normal_y>::value &&
                    pcl::traits::has_field<T, pcl::fields::normal_z>::value) {
        std::cerr << "    normal = [" << pt.normal_x << ", "
                  << pt.normal_y << ", " << pt.normal_z << "]\n";
      }
      if constexpr (pcl::traits::has_field<T, pcl::fields::curvature>::value) {
        std::cerr << "    curvature = " << pt.curvature << "\n";
      }
      break;
    }
  }*/
}


// ----------------------------------------------------------------------------------------------------------------------------------------------

template <typename PointSource, typename PointTarget>
bool NanoGICP<PointSource, PointTarget>::calculateSourceCovariances() {                   // calculate source covariances if not set by user
  std::cout << "[NanoGICP] STANDARD calculateSourceCovariances() called!" << std::endl;
  std::shared_ptr<CovarianceList> source_covs = std::make_shared<CovarianceList>();
  std::shared_ptr<float> source_density = std::make_shared<float>();
  bool ret = calculate_covariances(input_, *source_kdtree_, *source_covs, *source_density);
  source_covs_ = source_covs;
  source_density_ = *source_density;
  return ret;
}

template <typename PointSource, typename PointTarget>
bool NanoGICP<PointSource, PointTarget>::calculateTargetCovariances() {
  std::cout << "[NanoGICP] calculateTargetCovariances() called!" << std::endl;
  std::shared_ptr<CovarianceList> target_covs = std::make_shared<CovarianceList>();
  std::shared_ptr<float> target_density = std::make_shared<float>();
  bool ret = calculate_covariances(target_, *target_kdtree_, *target_covs, *target_density);
  target_covs_ = target_covs;
  target_density_ = *target_density;
  return ret;
}

// Used in DLIO.
template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::computeTransformation(PointCloudSource& output, const Matrix4& guess) {
  std::cout << "[NanoGICP] source_covs_ ptr=" << (void*)source_covs_.get()
            << " size=" << (source_covs_ ? source_covs_->size() : 0)
            << " input_size=" << (input_ ? input_->size() : -1) << std::endl;
  std::cout << "COMPUTE" << std::endl;
  
  

  if (source_covs_ == nullptr || source_covs_->size() != input_->size()) {    // Never called
    std::cout << "COMPUTE: CSC" << std::endl;
    calculateSourceCovariances();
    //retrieveSourceCovariancesFromROSMsg();
  }
  /*std::cout << "[NanoGICP] target_covs_ ptr=" << (void*)target_covs_.get()
            << " size=" << (target_covs_ ? target_covs_->size() : 0)
            << " target_size=" << (target_ ? target_->size() : -1) << std::endl;
  */
  if (target_covs_ == nullptr || target_covs_->size() != target_->size()) {   // Never called
    std::cout << "COMPUTE: CTC" << std::endl;
    calculateTargetCovariances();
    //retrieveSourceCovariancesFromROSMsg();
  }
  std::cerr << "\n=== BT: NanoGICP::computeTransformation ===\n";
  print_bt();
  LsqRegistration<PointSource, PointTarget>::computeTransformation(output, guess);
}

template <typename PointSource, typename PointTarget>
void NanoGICP<PointSource, PointTarget>::update_correspondences(const Eigen::Isometry3d& trans) {
  assert(source_covs_ != nullptr && source_covs_->size() == input_->size());
  assert(target_covs_ != nullptr && target_covs_->size() == target_->size());

  Eigen::Isometry3f trans_f = trans.cast<float>();

  correspondences_.resize(input_->size());
  sq_distances_.resize(input_->size());
  mahalanobis_.resize(input_->size());

  std::vector<int> k_indices(1);
  std::vector<float> k_sq_dists(1);

#pragma omp parallel for num_threads(num_threads_) firstprivate(k_indices, k_sq_dists) schedule(guided, 8)
  for (int i = 0; i < input_->size(); i++) {
    PointTarget pt;
    pt.getVector4fMap() = trans_f * input_->at(i).getVector4fMap();

    target_kdtree_->nearestKSearch(pt, 1, k_indices, k_sq_dists);

    sq_distances_[i] = k_sq_dists[0];
    correspondences_[i] = k_sq_dists[0] < corr_dist_threshold_ * corr_dist_threshold_ ? k_indices[0] : -1;

    if (correspondences_[i] < 0) {
      continue;
    }

    const int target_index = correspondences_[i];
    const auto& cov_A = (*source_covs_)[i];
    const auto& cov_B = (*target_covs_)[target_index];

    Eigen::Matrix4d RCR = cov_B + trans.matrix() * cov_A * trans.matrix().transpose();
    RCR(3, 3) = 1.0;

    mahalanobis_[i] = RCR.inverse();
    mahalanobis_[i](3, 3) = 0.0f;
  }

  num_correspondences = std::count_if(correspondences_.begin(), correspondences_.end(), [](int c){return c > 0;});
}

template <typename PointSource, typename PointTarget>
double NanoGICP<PointSource, PointTarget>::linearize(const Eigen::Isometry3d& trans, Eigen::Matrix<double, 6, 6>* H, Eigen::Matrix<double, 6, 1>* b) {
  update_correspondences(trans);

  double sum_errors = 0.0;
  std::vector<Eigen::Matrix<double, 6, 6>, Eigen::aligned_allocator<Eigen::Matrix<double, 6, 6>>> Hs(num_threads_);
  std::vector<Eigen::Matrix<double, 6, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 6, 1>>> bs(num_threads_);
  for (int i = 0; i < num_threads_; i++) {
    Hs[i].setZero();
    bs[i].setZero();
  }

#pragma omp parallel for num_threads(num_threads_) reduction(+ : sum_errors) schedule(guided, 8)
  for (int i = 0; i < input_->size(); i++) {
    int target_index = correspondences_[i];
    if (target_index < 0) {
      continue;
    }

    const Eigen::Vector4d mean_A = input_->at(i).getVector4fMap().template cast<double>();

    const Eigen::Vector4d mean_B = target_->at(target_index).getVector4fMap().template cast<double>();

    const Eigen::Vector4d transed_mean_A = trans * mean_A;
    const Eigen::Vector4d error = mean_B - transed_mean_A;

    sum_errors += error.transpose() * mahalanobis_[i] * error;

    if (H == nullptr || b == nullptr) {
      continue;
    }

    Eigen::Matrix<double, 4, 6> dtdx0 = Eigen::Matrix<double, 4, 6>::Zero();
    dtdx0.block<3, 3>(0, 0) = skewd(transed_mean_A.head<3>());
    dtdx0.block<3, 3>(0, 3) = -Eigen::Matrix3d::Identity();

    Eigen::Matrix<double, 4, 6> jlossexp = dtdx0;

    Eigen::Matrix<double, 6, 6> Hi = jlossexp.transpose() * mahalanobis_[i] * jlossexp;
    Eigen::Matrix<double, 6, 1> bi = jlossexp.transpose() * mahalanobis_[i] * error;

    Hs[omp_get_thread_num()] += Hi;
    bs[omp_get_thread_num()] += bi;
  }

  if (H && b) {
    H->setZero();
    b->setZero();
    for (int i = 0; i < num_threads_; i++) {
      (*H) += Hs[i];
      (*b) += bs[i];
    }
  }

  return sum_errors;
}

template <typename PointSource, typename PointTarget>
double NanoGICP<PointSource, PointTarget>::compute_error(const Eigen::Isometry3d& trans) {
  double sum_errors = 0.0;

#pragma omp parallel for num_threads(num_threads_) reduction(+ : sum_errors) schedule(guided, 8)
  for (int i = 0; i < input_->size(); i++) {
    int target_index = correspondences_[i];
    if (target_index < 0) {
      continue;
    }

    const Eigen::Vector4d mean_A = input_->at(i).getVector4fMap().template cast<double>();

    const Eigen::Vector4d mean_B = target_->at(target_index).getVector4fMap().template cast<double>();

    const Eigen::Vector4d transed_mean_A = trans * mean_A;
    const Eigen::Vector4d error = mean_B - transed_mean_A;

    sum_errors += error.transpose() * mahalanobis_[i] * error;
  }

  return sum_errors;
}

template <typename PointSource, typename PointTarget>
template <typename PointT>
bool NanoGICP<PointSource, PointTarget>::calculate_covariances(           // calculate covariances for a point cloud
  const typename pcl::PointCloud<PointT>::ConstPtr& cloud,
  const nanoflann::KdTreeFLANN<PointT>& kdtree,
  CovarianceList& covariances,
  float& density) {

  covariances.resize(cloud->size());
  float sum_k_sq_distances = 0.0;

#pragma omp parallel for num_threads(num_threads_) schedule(guided, 8) reduction(+:sum_k_sq_distances)
  for (int i = 0; i < cloud->size(); i++) {
    std::vector<int> k_indices;
    std::vector<float> k_sq_distances;
    kdtree.nearestKSearch(cloud->at(i), k_correspondences_, k_indices, k_sq_distances);

    const int normalization = ((k_correspondences_ - 1) * (2 + k_correspondences_)) / 2;
    sum_k_sq_distances += std::accumulate(k_sq_distances.begin()+1, k_sq_distances.end(), 0.0) / normalization;

    Eigen::Matrix<double, 4, -1> neighbors(4, k_correspondences_);
    for (int j = 0; j < k_indices.size(); j++) {
      neighbors.col(j) = cloud->at(k_indices[j]).getVector4fMap().template cast<double>();
    }

    neighbors.colwise() -= neighbors.rowwise().mean().eval();
    Eigen::Matrix4d cov = neighbors * neighbors.transpose() / k_correspondences_;

    if (regularization_method_ == RegularizationMethod::NONE) {
      covariances[i] = cov;
    } else if (regularization_method_ == RegularizationMethod::FROBENIUS) {
      double lambda = 1e-3;
      Eigen::Matrix3d C = cov.block<3, 3>(0, 0).cast<double>() + lambda * Eigen::Matrix3d::Identity();
      Eigen::Matrix3d C_inv = C.inverse();
      covariances[i].setZero();
      covariances[i].template block<3, 3>(0, 0) = (C_inv / C_inv.norm()).inverse();
    } else {
      Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov.block<3, 3>(0, 0), Eigen::ComputeFullU | Eigen::ComputeFullV);
      Eigen::Vector3d values;

      switch (regularization_method_) {
        default:
          std::cerr << "here must not be reached" << std::endl;
          abort();
        case RegularizationMethod::PLANE:
          values = Eigen::Vector3d(1, 1, 1e-3);
          break;
        case RegularizationMethod::MIN_EIG:
          values = svd.singularValues().array().max(1e-3);
          break;
        case RegularizationMethod::NORMALIZED_MIN_EIG:
          values = svd.singularValues() / svd.singularValues().maxCoeff();
          values = values.array().max(1e-3);
          break;
      }

      covariances[i].setZero();
      covariances[i].template block<3, 3>(0, 0) = svd.matrixU() * values.asDiagonal() * svd.matrixV().transpose();
    }
  }

  density = sum_k_sq_distances / cloud->size();

  return true;
}

}  // namespace nano_gicp
