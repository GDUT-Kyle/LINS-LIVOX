#ifndef _INCLUDE_MYIMU_H
#define _INCLUDE_MYIMU_H
#include <eigen3/Eigen/Dense>
#include <sensor_msgs/Imu.h>

class myImu
{
public:
    double timestamp_s;
    Eigen::Quaternionf pose;
    Eigen::Vector3f acc;
    Eigen::Vector3f gyr;
public:
    myImu(const sensor_msgs::Imu::ConstPtr& imuIn)
    {
        timestamp_s = imuIn->header.stamp.toSec();

        pose.x() = imuIn->orientation.x;
        pose.y() = imuIn->orientation.y;
        pose.z() = imuIn->orientation.z;
        pose.w() = imuIn->orientation.w;

        // std::cout<<"pose : "<<pose.toRotationMatrix().eulerAngles(0, 1, 2).transpose()<<std::endl;

        acc.x() = imuIn->linear_acceleration.x;
        acc.y() = imuIn->linear_acceleration.y;
        acc.z() = imuIn->linear_acceleration.z;

        gyr.x() = imuIn->angular_velocity.x;
        gyr.y() = imuIn->angular_velocity.y;
        gyr.z() = imuIn->angular_velocity.z;
    }
    myImu(){}
    ~myImu(){}
};

#endif