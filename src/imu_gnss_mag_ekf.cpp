#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/MagneticField.h>
#include <sensor_msgs/NavSatFix.h>

#include <deque>
#include <fstream>
#include <iostream>
#include <mutex>

#include "common/view.hpp"
#include "estimator/ekf.hpp"
#include "sensor/gnss_mag.hpp"
#include "sensor/imu.hpp"
#include "wmm/mag.h"

namespace cg {

ANGULAR_ERROR State::kAngError = ANGULAR_ERROR::LOCAL_ANGULAR_ERROR;

class FusionNode {
 public:
  FusionNode(ros::NodeHandle &nh) : viewer_(nh) {
    double acc_n, gyr_n, acc_w, gyr_w;
    nh.param("acc_noise", acc_n, 1e-2);
    nh.param("gyr_noise", gyr_n, 1e-4);
    nh.param("acc_bias_noise", acc_w, 1e-6);
    nh.param("gyr_bias_noise", gyr_w, 1e-8);

    const double sigma_pv = 10;
    const double sigma_rp = 10 * kDegreeToRadian;
    const double sigma_yaw = 100 * kDegreeToRadian;

    ekf_ptr_ = std::make_unique<EKF>();
    ekf_ptr_->state_ptr_->set_cov(sigma_pv, sigma_pv, sigma_rp, sigma_yaw, 0.02, 0.02);
    ekf_ptr_->predictor_ptr_ = std::make_shared<IMU>(ekf_ptr_->state_ptr_, acc_n, gyr_n, acc_w, gyr_w);
    ekf_ptr_->observer_ptr_ = std::make_shared<GNSS_MAG>();

    std::string topic_imu = "/livox/imu";
    // std::string topic_imu = "/mavros/imu/data_raw";
    std::string topic_gps = "/ublox_driver/receiver_lla";
    std::string topic_mag = "/mavros/imu/mag";
    wmm_cof_path_ = "/home/peiwen/GIM_ws/src/imu_x_fusion/config/WMM.COF";

    imu_sub_ = nh.subscribe<sensor_msgs::Imu>(topic_imu, 10, boost::bind(&FusionNode::imu_callback, this, _1));

    gps_sub_ = nh.subscribe(topic_gps, 10, &FusionNode::gps_callback, this);
    
    mag_sub_ = nh.subscribe(topic_mag, 10, &FusionNode::mag_callback, this);

    // log files
    file_gps_.open("fusion_gps.csv");
    file_state_.open("fusion_state.csv");
  }

  ~FusionNode() {
    if (file_gps_.is_open()) file_gps_.close();
    if (file_state_.is_open()) file_state_.close();
  }

  void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg) {
    Eigen::Vector3d acc, gyr;
    double g = 9.81;
    acc[0] = imu_msg->linear_acceleration.x *g;
    acc[1] = imu_msg->linear_acceleration.y *g;
    acc[2] = imu_msg->linear_acceleration.z *g;

    gyr[0] = imu_msg->angular_velocity.x;
    gyr[1] = imu_msg->angular_velocity.y;
    gyr[2] = imu_msg->angular_velocity.z;

    ekf_ptr_->predict(std::make_shared<ImuData>(imu_msg->header.stamp.toSec(), acc, gyr));
  }

  void mag_callback(const sensor_msgs::MagneticField::ConstPtr& msg) {
    std::lock_guard<std::mutex> lock(mag_mutex_);
      // Add the new message to the deque
    Eigen::Vector3d magnetic, magnetic_bias;
    double scale = 2.0;
    
    magnetic << msg->magnetic_field.x* 1e6, msg->magnetic_field.y* 1e6, msg->magnetic_field.z* 1e6;
    magnetic_bias << 0 ,0 ,0;
    magnetic = scale* magnetic + magnetic_bias;

    Eigen::Vector4d t_mag;
    t_mag[0] = msg->header.stamp.toSec();
    t_mag.tail(3) = magnetic;

    mag_buf_.push_back(t_mag); // Remove the oldest message if the cache size limit is exceeded 
    if (mag_buf_.size() > mag_buf_size) { 
      mag_buf_.pop_front(); 
    }
    // Log the most recent message (for demonstration purposes)
    ROS_INFO("Cached Magnetic Field: [x: %f, y: %f, z: %f]", magnetic[0], magnetic[1], magnetic[2]);
  }

  Eigen::Vector3d interpolate_mag(double t_gps);

  void gps_callback(const sensor_msgs::NavSatFixConstPtr &gps_msg);

 private:
  ros::Subscriber imu_sub_;
  ros::Subscriber gps_sub_;
  ros::Subscriber mag_sub_;

  std::deque<Eigen::Vector4d> mag_buf_;
  const std::size_t mag_buf_size = 100;
  std::mutex mag_mutex_;
  bool first_mag_ = true;
  Eigen::Vector3d Mag_ENU;

  EKFPtr ekf_ptr_;
  Viewer viewer_;

  std::ofstream file_gps_;
  std::ofstream file_state_;
  std::string wmm_cof_path_;

};

void FusionNode::gps_callback(const sensor_msgs::NavSatFixConstPtr &gps_msg) {
  if (gps_msg->status.status != 3) {
    printf("[cggos %s] ERROR: Bad GPS Message!!!\n", __FUNCTION__);
    return;
  }

  GpsMagData::Ptr gps_mag_data_ptr = std::make_shared<GpsMagData>();
  gps_mag_data_ptr->timestamp = gps_msg->header.stamp.toSec() - 18.0 + 0.08;
  gps_mag_data_ptr->lla[0] = gps_msg->latitude;
  gps_mag_data_ptr->lla[1] = gps_msg->longitude;
  gps_mag_data_ptr->lla[2] = gps_msg->altitude;

  if(first_mag_)
  {
    first_mag_ = false;
    if(!calc_local_mag_field(gps_mag_data_ptr->lla, Mag_ENU, wmm_cof_path_))
    {
      std::cerr << "Error calc_local_mag_field. " << std::endl;
      return; 
    }
    std::cout << "[ Mag ENU ]: " << Mag_ENU.transpose() << std::endl;
  }
  // try { 
  //   gps_mag_data_ptr->mag = interpolate_mag(gps_mag_data_ptr->timestamp);
  // } catch (const std::exception& e) { 
  //   std::cerr << "Error: " << e.what() << std::endl; 
  //   return;
  // }

  gps_mag_data_ptr->mag = mag_buf_.back().tail(3);

  std::cout << "interpolate_mag"  << std::endl;
  // gps_mag_data_ptr->cov = Eigen::Map<const Eigen::Matrix3d>(gps_msg->position_covariance.data());
  Eigen::Matrix<double, kMeasDim, kMeasDim> cov;
  cov.setZero();
  Eigen::VectorXd vec6d;
  vec6d = Eigen::MatrixXd::Zero(6, 1);
  vec6d << 1, 1, 1, 1, 1, 1;
  cov.diagonal() = vec6d;
  std::cout <<  vec6d.transpose() << std::endl;
  
  gps_mag_data_ptr->cov = cov;
  std::cout <<  "Setting gps mag cov matrix: " << cov << std::endl;

  if (!ekf_ptr_->predictor_ptr_->inited_) {
    if (!ekf_ptr_->predictor_ptr_->init(gps_mag_data_ptr->timestamp)) return;

    std::dynamic_pointer_cast<GNSS_MAG>(ekf_ptr_->observer_ptr_)->set_params(gps_mag_data_ptr, Eigen::Vector3d::Zero(), Mag_ENU);

    printf("[cggos %s] System initialized.\n", __FUNCTION__);

    return;
  }

  std::cout << "---------------------" << std::endl;

  const Eigen::Isometry3d &Twb = ekf_ptr_->state_ptr_->pose();
  const auto &p_G_Gps_Mag = std::dynamic_pointer_cast<GNSS_MAG>(ekf_ptr_->observer_ptr_)->g2l_mag(gps_mag_data_ptr);

  std::cout << "p_G_Gps_Mag: " << p_G_Gps_Mag.transpose() << std::endl;

  const auto &residual = ekf_ptr_->observer_ptr_->measurement_residual(Twb.matrix(), p_G_Gps_Mag);

  std::cout << "res: " << residual.transpose() << std::endl;

  const auto &H = ekf_ptr_->observer_ptr_->measurement_jacobian(Twb.matrix(), p_G_Gps_Mag);

  Eigen::Matrix<double, kStateDim, kMeasDim> K;
  const Eigen::Matrix<double, kMeasDim, kMeasDim> &R = gps_mag_data_ptr->cov;
  ekf_ptr_->update_K(H, R, K);
  ekf_ptr_->update_P(H, R, K);
  *ekf_ptr_->state_ptr_ = *ekf_ptr_->state_ptr_ + K * residual;

  std::cout << "acc bias: " << ekf_ptr_->state_ptr_->acc_bias.transpose() << std::endl;
  std::cout << "gyr bias: " << ekf_ptr_->state_ptr_->gyr_bias.transpose() << std::endl;
  std::cout << "---------------------" << std::endl;

  // save data
  {
    viewer_.publish_gnss(*ekf_ptr_->state_ptr_);

    // save state p q lla
    const auto &lla = std::dynamic_pointer_cast<GNSS_MAG>(ekf_ptr_->observer_ptr_)->l2g(ekf_ptr_->state_ptr_->p_wb_);

    const Eigen::Quaterniond q_GI(ekf_ptr_->state_ptr_->Rwb_);
    file_state_ << std::fixed << std::setprecision(15) << ekf_ptr_->state_ptr_->timestamp << ", "
                << ekf_ptr_->state_ptr_->p_wb_[0] << ", " << ekf_ptr_->state_ptr_->p_wb_[1] << ", "
                << ekf_ptr_->state_ptr_->p_wb_[2] << ", " << q_GI.x() << ", " << q_GI.y() << ", " << q_GI.z() << ", "
                << q_GI.w() << ", " << lla[0] << ", " << lla[1] << ", " << lla[2] << std::endl;

    file_gps_ << std::fixed << std::setprecision(15) << gps_mag_data_ptr->timestamp << ", " << gps_mag_data_ptr->lla[0] << ", "
              << gps_mag_data_ptr->lla[1] << ", " << gps_mag_data_ptr->lla[2] << std::endl;
  }
}

Eigen::Vector3d FusionNode::interpolate_mag(double t_gps) {
  std::lock_guard<std::mutex> lock(mag_mutex_);

  // Ensure there's data to interpolate 
  if (mag_buf_.size() < 2) { 
    throw std::runtime_error("Not enough magnetic field measurements for interpolation."); 
  }

  Eigen::Vector4d prev, next;
  bool found = false;

  for (size_t i = 1; i < mag_buf_.size(); ++i) {
    if (mag_buf_[i-1][0] <= t_gps && mag_buf_[i][0] > t_gps) {
      prev = mag_buf_[i-1];
      next = mag_buf_[i];
      found = true;
      break;
    }
  }

  if (!found) {
    throw std::runtime_error("Unable to find adjacent magnetic field measurements for interpolation.");
  }

  double t0 = prev[0];
  double t1 = next[0];
  double ratio = (t_gps - t0) / (t1 - t0);

  Eigen::Vector3d mag_interpolated = (1 - ratio) * prev.tail<3>() + ratio * next.tail<3>();

  ROS_INFO("Interpolated Magnetic Field: [x: %f, y: %f, z: %f]", mag_interpolated[0], mag_interpolated[1], mag_interpolated[2]);

  // Remove all data before t_gps 
  while (!mag_buf_.empty() && mag_buf_.front()[0] <= t_gps) { 
    mag_buf_.pop_front(); 
  } 

  return mag_interpolated;
}

}  // namespace cg

int main(int argc, char **argv) {
  ros::init(argc, argv, "imu_gnss_mag_fusion");

  ros::NodeHandle nh;
  cg::FusionNode fusion_node(nh);

  ros::spin();

  return 0;
}


