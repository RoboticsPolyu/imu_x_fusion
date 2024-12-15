#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <GeographicLib/LocalCartesian.hpp>
#include <memory>

#include "fusion/observer.hpp"
#include "wmm/GeomagnetismLibrary.hpp"

namespace cg {

constexpr int kMeasDim = 6;

struct GpsMagData {
  double timestamp;

  Eigen::Vector3d lla;  // Latitude in degree, longitude in degree, and altitude in meter
  Eigen::Matrix3d mag;  // Mag in uT
  
  Eigen::Matrix6d cov;  // Covariance in m^2

  using Ptr = std::shared_ptr<GpsMagData>;
  using ConstPtr = std::shared_ptr<const GpsMagData>;
};

class GNSS_MAG : public Observer {
 public:
  using Ptr = std::shared_ptr<GNSS_MAG>;

  GNSS_MAG() = default;

  virtual ~GNSS_MAG() {}

  void set_params(GpsMagData::ConstPtr gps_data_ptr, const Eigen::Vector3d &I_p_Gps = Eigen::Vector3d::Zero(), const Eigen::Vector3d &Mag_ENU) {
    init_lla_ = gps_data_ptr->lla;
    I_p_Gps_ = I_p_Gps;
    Mag_ENU_ = Mag_ENU;
  }

  virtual Eigen::MatrixXd measurement_function(const Eigen::MatrixXd &mat_x) {
    Eigen::Isometry3d Twb;
    Twb.matrix() = mat_x;

    Eigen::Matrix<double, kMeasDim, 1> residual;
    residual.topRows(3) = Twb * I_p_Gps_;
    residual.bottomRows(3) = Twb.rotation().transpose()* Mag_ENU_;
  }

  virtual Eigen::MatrixXd measurement_residual(const Eigen::MatrixXd &mat_x, const Eigen::MatrixXd &mat_z) {
    return mat_z - measurement_function(mat_x);
  }

  virtual Eigen::MatrixXd measurement_jacobian(const Eigen::MatrixXd &mat_x, const Eigen::MatrixXd &mat_z) {
    Eigen::Isometry3d Twb;
    Twb.matrix() = mat_x;

    Eigen::Matrix<double, kMeasDim, kStateDim> H;
    H.setZero();
    H.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    H.block<3, 3>(0, 6) = -Twb.linear() * Utils::skew_matrix(I_p_Gps_);
    H.block<3, 3>(3, 6) = -Utils::skew_matrix(Twb.rotation().transpose()* Mag_ENU_);
    return H;
  }

  virtual void check_jacobian(const Eigen::MatrixXd &mat_x, const Eigen::MatrixXd &mat_z) {}

  /**
   * @brief global to local coordinate, convert WGS84 to ENU frame
   *
   * @param gps_data_ptr
   * @return Eigen::Vector3d
   */
  Eigen::Vector3d g2l(GpsMagData::ConstPtr gps_data_ptr) {
    Eigen::Vector3d p_G_Gps;
    GNSS_MAG::lla2enu(init_lla_, gps_data_ptr->lla, &p_G_Gps);
    return p_G_Gps;
  }

  Eigen::Vector6d g2l_mag(GpsMagData::ConstPtr gps_data_ptr) {
    Eigen::Vector3d p_G_Gps;
    GNSS_MAG::lla2enu(init_lla_, gps_data_ptr->lla, &p_G_Gps);
    Eigen::Vector6d p_G_Gps_Mag;
    p_G_Gps_Mag.head(3) = p_G_Gps;
    p_G_Gps_Mag.tail(3) = gps_data_ptr->mag;
    return p_G_Gps_Mag;
  }

  /**
   * @brief local to glocal coordinate, convert ENU to WGS84 lla
   *
   * @param p_wb
   * @return Eigen::Vector3d
   */
  Eigen::Vector3d l2g(const Eigen::Vector3d &p_wb) {
    Eigen::Vector3d lla;
    GNSS_MAG::enu2lla(init_lla_, p_wb, &lla);
    return lla;
  }

  static inline void lla2enu(const Eigen::Vector3d &init_lla,
                             const Eigen::Vector3d &point_lla,
                             Eigen::Vector3d *point_enu) {
    static GeographicLib::LocalCartesian local_cartesian;
    local_cartesian.Reset(init_lla(0), init_lla(1), init_lla(2));
    local_cartesian.Forward(
        point_lla(0), point_lla(1), point_lla(2), point_enu->data()[0], point_enu->data()[1], point_enu->data()[2]);
  }

  static inline void enu2lla(const Eigen::Vector3d &init_lla,
                             const Eigen::Vector3d &point_enu,
                             Eigen::Vector3d *point_lla) {
    static GeographicLib::LocalCartesian local_cartesian;
    local_cartesian.Reset(init_lla(0), init_lla(1), init_lla(2));
    local_cartesian.Reverse(
        point_enu(0), point_enu(1), point_enu(2), point_lla->data()[0], point_lla->data()[1], point_lla->data()[2]);
  }

 private:
  Eigen::Vector3d init_lla_;
  Eigen::Vector3d I_p_Gps_  = Eigen::Vector3d::Zero(); // GPS in the Inertal frame (extrinsic parameters)
  Eigen::Vector3d Mag_ENU_ = Eigen::Vector3d::Zero(); // Ref Mag Value (T) based on inital GPS in ENU
  Eigen::Vector3d I_mag_    = Eigen::Vector3d::Zero(); // Mag Measurement (T) in ENU 
  
};

}  // namespace cg
