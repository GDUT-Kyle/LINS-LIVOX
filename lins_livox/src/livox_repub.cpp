#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include "livox_ros_driver/CustomMsg.h"
#include "lins_livox/common.h"

pcl::PointCloud<pcl::PointXYZRGB>::Ptr laserCloudFullResColor(
    new pcl::PointCloud<pcl::PointXYZRGB>());

Eigen::Affine3f Ext_Livox = Eigen::Affine3f::Identity();

ros::Subscriber subImu;
ros::Publisher pub_pcl_out0, pub_pcl_out1, pubLaserCloudFullRes;
uint64_t TO_MERGE_CNT = 1; 
constexpr bool b_dbg_line = false;
bool initialImu = false;
// 创建一个循环队列用于存储雷达帧
std::vector<livox_ros_driver::CustomMsgConstPtr> livox_data;

void RGBpointAssociateToMap(PointType const *const pi,
                            pcl::PointXYZRGB *const po) {
  Eigen::Vector3d point_curr(pi->x, pi->y, pi->z);
  po->x = point_curr.x();
  po->y = point_curr.y();
  po->z = point_curr.z();
  int reflection_map = pi->curvature * 1000;
  if (reflection_map < 30) {
    int green = (reflection_map * 255 / 30);
    po->r = 0;
    po->g = green & 0xff;
    po->b = 0xff;
  } else if (reflection_map < 90) {
    int blue = (((90 - reflection_map) * 255) / 60);
    po->r = 0x0;
    po->g = 0xff;
    po->b = blue & 0xff;
  } else if (reflection_map < 150) {
    int red = ((reflection_map - 90) * 255 / 60);
    po->r = red & 0xff;
    po->g = 0xff;
    po->b = 0x0;
  } else {
    int green = (((255 - reflection_map) * 255) / (255 - 150));
    po->r = 0xff;
    po->g = green & 0xff;
    po->b = 0;
  }
}

void LivoxMsgCbk1(const livox_ros_driver::CustomMsgConstPtr& livox_msg_in) {
  livox_data.push_back(livox_msg_in);
  // 第一帧则跳过
  if (livox_data.size() < TO_MERGE_CNT) return;

  pcl::PointCloud<PointType> pcl_in;

  // 遍历队列
  for (size_t j = 0; j < livox_data.size(); j++) {
    // 通过引用，方便操作每一帧
    auto& livox_msg = livox_data[j];
    // 获取该帧最后一个点的相对时间
    auto time_end = livox_msg->points.back().offset_time;
    // 重新组织成PCL的点云
    for (unsigned int i = 0; i < livox_msg->point_num; ++i) {
      PointType pt;
      pt.x = livox_msg->points[i].x;
      pt.y = livox_msg->points[i].y;
      pt.z = livox_msg->points[i].z;
//      if (pt.z < -0.3) continue; // delete some outliers (our Horizon's assembly height is 0.3 meters)
      float s = livox_msg->points[i].offset_time / (float)time_end;
//       ROS_INFO("_s-------- %.6f ",s);
      // 线数——整数，时间偏移——小数
      pt.intensity = livox_msg->points[i].line + s*0.1; // The integer part is line number and the decimal part is timestamp
//      ROS_INFO("intensity-------- %.6f ",pt.intensity);
      pt.curvature = livox_msg->points[i].reflectivity * 0.001;
      // ROS_INFO("pt.curvature-------- %.3f ",pt.curvature);
      pcl_in.push_back(pt);
    }
  }

  /// timebase 5ms ~ 50000000, so 10 ~ 1ns
  pcl::transformPointCloud(pcl_in, pcl_in, Ext_Livox);

  // 最新一帧的时间戳
  unsigned long timebase_ns = livox_data[0]->timebase;
  ros::Time timestamp;
  timestamp.fromNSec(timebase_ns);

  //   ROS_INFO("livox1 republish %u points at time %f buf size %ld",
  //   pcl_in.size(),
  //           timestamp.toSec(), livox_data.size());

  sensor_msgs::PointCloud2 pcl_ros_msg;
  pcl::toROSMsg(pcl_in, pcl_ros_msg);
  pcl_ros_msg.header.stamp.fromNSec(timebase_ns);
  pcl_ros_msg.header.frame_id = "/livox";
  pub_pcl_out1.publish(pcl_ros_msg);
  livox_data.clear();

  if(pubLaserCloudFullRes.getNumSubscribers() != 0)
  {
    laserCloudFullResColor->clear();
    int laserCloudFullResNum = pcl_in.points.size();
    for (int i = 0; i < laserCloudFullResNum; i++) {
      pcl::PointXYZRGB temp_point;
      RGBpointAssociateToMap(&pcl_in.points[i], &temp_point);
      laserCloudFullResColor->push_back(temp_point);
    }
    sensor_msgs::PointCloud2 laserCloudFullRes3;
    pcl::toROSMsg(*laserCloudFullResColor, laserCloudFullRes3);
    laserCloudFullRes3.header.stamp.fromNSec(timebase_ns);
    laserCloudFullRes3.header.frame_id = "/livox";
    pubLaserCloudFullRes.publish(laserCloudFullRes3);
  }
}

// 订阅IMU
void imuHandler(const sensor_msgs::Imu::ConstPtr& imuIn)
{
    #ifdef INITIAL_BY_IMU
    if(!initialImu)
    {
        // float imupitch_ = atan2(-imuIn->linear_acceleration.x, 
        //                   sqrt(imuIn->linear_acceleration.z*imuIn->linear_acceleration.z + 
        //                   imuIn->linear_acceleration.y*imuIn->linear_acceleration.y));
        int signAccZ;
        if(imuIn->linear_acceleration.z >= 0) signAccZ = 1;
        else signAccZ = -1;
        float imupitch_ = -signAccZ * asin(imuIn->linear_acceleration.x);
        float imuroll_ = signAccZ * asin(imuIn->linear_acceleration.y);
        // float imuroll_ = atan2(imuIn->linear_acceleration.y, imuIn->linear_acceleration.z);
        Eigen::AngleAxisf imuPitch = Eigen::AngleAxisf(imupitch_, Eigen::Vector3f::UnitY());
        Eigen::AngleAxisf imuRoll = Eigen::AngleAxisf(imuroll_, Eigen::Vector3f::UnitX());
        Ext_Livox.rotate(imuRoll * imuPitch);
        // Ext_Livox.pretranslate();
        initialImu = true;
    }
    #endif
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "livox_repub");
  ros::NodeHandle nh;

  ROS_INFO("start livox_repub");

  #ifndef INITIAL_BY_IMU
  Eigen::AngleAxisf imuPitch = Eigen::AngleAxisf(livox_mount_pitch, Eigen::Vector3f::UnitY());
  Eigen::AngleAxisf imuRoll = Eigen::AngleAxisf(livox_mount_roll, Eigen::Vector3f::UnitX());
  Ext_Livox.rotate(imuRoll * imuPitch);
  #endif

  ros::Subscriber sub_livox_msg1 = nh.subscribe<livox_ros_driver::CustomMsg>(
      "/livox/lidar", 100, LivoxMsgCbk1);
  pub_pcl_out1 = nh.advertise<sensor_msgs::PointCloud2>("/livox_pcl0", 100);

  pubLaserCloudFullRes =
      nh.advertise<sensor_msgs::PointCloud2>("/color_livox_pcl0", 100);

  subImu = nh.subscribe<sensor_msgs::Imu>(imuTopic, 1, imuHandler);

  ros::spin();
}
