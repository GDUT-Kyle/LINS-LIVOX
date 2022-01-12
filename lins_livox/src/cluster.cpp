/*
 * Cluster.cpp
 *
 *  Created on: Oct 19, 2016
 *      Author: Ne0
 */

#include "lins_livox/cluster.h"
// #include "lins_livox/common.h"

Cluster::Cluster()
{
  valid_cluster_ = true;
}

geometry_msgs::PolygonStamped Cluster::GetPolygon()
{
  return polygon_;
}

// jsk_recognition_msgs::BoundingBox Cluster::GetBoundingBox()
// {
//   return bounding_box_;
// }

pcl::PointCloud<pcl::PointXYZRGB>::Ptr Cluster::GetCloud()
{
  return pointcloud_;
}

PointType Cluster::GetMinPoint()
{
  return min_point_;
}

PointType Cluster::GetMaxPoint()
{
  return max_point_;
}

PointType Cluster::GetCentroid()
{
  return centroid_;
}

PointType Cluster::GetAveragePoint()
{
  return average_point_;
}

double Cluster::GetOrientationAngle()
{
  return orientation_angle_;
}

Eigen::Matrix3f Cluster::GetEigenVectors()
{
  return eigen_vectors_;
}

Eigen::Vector3f Cluster::GetEigenValues()
{
  return eigen_values_;
}

void Cluster::SetCloud(const pcl::PointCloud<PointType>::Ptr in_origin_cloud_ptr,
                       const std::vector<int>& in_cluster_indices, std_msgs::Header in_ros_header, int in_id, int in_r,
                       int in_g, int in_b, std::string in_label, bool in_estimate_pose)
{
  label_ = in_label;
  id_ = in_id;
  r_ = in_r;
  g_ = in_g;
  b_ = in_b;
  // extract pointcloud using the indices
  // calculate min and max points
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr current_cluster(new pcl::PointCloud<pcl::PointXYZRGB>);
  float min_x = std::numeric_limits<float>::max();
  float max_x = -std::numeric_limits<float>::max();
  float min_y = std::numeric_limits<float>::max();
  float max_y = -std::numeric_limits<float>::max();
  float min_z = std::numeric_limits<float>::max();
  float max_z = -std::numeric_limits<float>::max();
  float average_x = 0, average_y = 0, average_z = 0;

  // 遍历这个聚类中的所有索引
  for (auto pit = in_cluster_indices.begin(); pit != in_cluster_indices.end(); ++pit)
  {
    // fill new colored cluster point by point
    // 获取该点的原始坐标和颜色
    pcl::PointXYZRGB p;
    p.x = in_origin_cloud_ptr->points[*pit].x;
    p.y = in_origin_cloud_ptr->points[*pit].y;
    p.z = in_origin_cloud_ptr->points[*pit].z;
    p.r = in_r;
    p.g = in_g;
    p.b = in_b;

    // 计算质心
    average_x += p.x;
    average_y += p.y;
    average_z += p.z;
    centroid_.x += p.x;
    centroid_.y += p.y;
    centroid_.z += p.z;
    current_cluster->points.push_back(p);

    // 计算边框
    if (p.x < min_x)
      min_x = p.x;
    if (p.y < min_y)
      min_y = p.y;
    if (p.z < min_z)
      min_z = p.z;
    if (p.x > max_x)
      max_x = p.x;
    if (p.y > max_y)
      max_y = p.y;
    if (p.z > max_z)
      max_z = p.z;
  }
  // min, max points
  min_point_.x = min_x;
  min_point_.y = min_y;
  min_point_.z = min_z;
  max_point_.x = max_x;
  max_point_.y = max_y;
  max_point_.z = max_z;

  // calculate centroid, average
  if (in_cluster_indices.size() > 0)
  {
    centroid_.x /= in_cluster_indices.size();
    centroid_.y /= in_cluster_indices.size();
    centroid_.z /= in_cluster_indices.size();

    average_x /= in_cluster_indices.size();
    average_y /= in_cluster_indices.size();
    average_z /= in_cluster_indices.size();
  }

  average_point_.x = average_x;
  average_point_.y = average_y;
  average_point_.z = average_z;

  // calculate bounding box
  length_ = max_point_.x - min_point_.x;
  width_ = max_point_.y - min_point_.y;
  height_ = max_point_.z - min_point_.z;


  // pose estimation
  double rz = 0;

  {
    std::vector<cv::Point2f> points;
    for (unsigned int i = 0; i < current_cluster->points.size(); i++)
    {
      cv::Point2f pt;
      pt.x = current_cluster->points[i].x;
      pt.y = current_cluster->points[i].y;
      points.push_back(pt);
    }

    std::vector<cv::Point2f> hull;
    // 寻找图像的凸包 https://www.cnblogs.com/jclian91/p/9728488.html
    cv::convexHull(points, hull);

    polygon_.header = in_ros_header;
    for (size_t i = 0; i < hull.size() + 1; i++)
    {
      geometry_msgs::Point32 point;
      point.x = hull[i % hull.size()].x;
      point.y = hull[i % hull.size()].y;
      point.z = min_point_.z;
      polygon_.polygon.points.push_back(point);
    }

    for (size_t i = 0; i < hull.size() + 1; i++)
    {
      geometry_msgs::Point32 point;
      point.x = hull[i % hull.size()].x;
      point.y = hull[i % hull.size()].y;
      point.z = max_point_.z;
      polygon_.polygon.points.push_back(point);
    }
  }

  current_cluster->width = current_cluster->points.size();
  current_cluster->height = 1;
  current_cluster->is_dense = true;


  valid_cluster_ = true;
  pointcloud_ = current_cluster;
}


bool Cluster::IsValid()
{
  return valid_cluster_;
}

void Cluster::SetValidity(bool in_valid)
{
  valid_cluster_ = in_valid;
}

int Cluster::GetId()
{
  return id_;
}

Cluster::~Cluster()
{
  // TODO Auto-generated destructor stub
}
