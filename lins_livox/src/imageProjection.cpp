#include <pcl_conversions/pcl_conversions.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include "livox_ros_driver/CustomMsg.h"
#include "lins_livox/common.h"
#include "lins_livox/cluster.h"

class ImageProjection{
private:
    ros::NodeHandle nh;

    ros::Subscriber subLaserCloud;
    
    ros::Publisher pubFullCloud;
    ros::Publisher pubFullInfoCloud;

    ros::Publisher pubGroundCloud;
    ros::Publisher pubSegmentedCloud;
    ros::Publisher pubSegmentedCloudPure;
    ros::Publisher pubSegmentedCloudInfo;
    // ros::Publisher pubOutlierCloud;
    ros::Publisher pubClusterCloud;

    ros::Publisher pubMarkerArray;

    ros::Publisher pubPlaneEquation;

    pcl::PointCloud<PointType>::Ptr laserCloudIn; //雷达直接传出的点云

    pcl::PointCloud<PointType>::Ptr fullCloud; //投影后的点云
    pcl::PointCloud<PointType>::Ptr fullInfoCloud; //整体的点云

    pcl::PointCloud<PointType>::Ptr groundCloud; //地面点云
    pcl::PointCloud<PointType>::Ptr segmentedCloud; //分割后的部分
    pcl::PointCloud<PointType>::Ptr segmentedCloudPure; //分割后的部分的几何信息
    // pcl::PointCloud<PointType>::Ptr outlierCloud; //在分割时出现的异常
    pcl::PointCloud<PointType>::Ptr groundCloudSeeds; //地面点云

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_clustered_cloud_ptr;

    #ifdef IS_CLUSTERS
    pcl::search::KdTree<PointType>::Ptr tree;
    // pcl::KdTreeFLANN<PointType>::Ptr tree;
    std::vector<pcl::PointCloud<PointType>> clusters;//保存分割后的所有类 每一类为一个点云
    // 欧式聚类对检测到的障碍物进行分组
	float clusterTolerance = 1.0;
	int minsize = 50;
	int maxsize = 3000;
	std::vector<pcl::PointIndices> clusterIndices;// 创建索引类型对象
	pcl::EuclideanClusterExtraction<PointType> ec; // 欧式聚类对象
    #endif


    PointType nanPoint;

    cv::Mat rangeMat;
    cv::Mat labelMat;
    cv::Mat groundMat;
    int labelCount;

    float startOrientation;
    float endOrientation;

    cloud_msgs::cloud_info segMsg;
    std_msgs::Header cloudHeader;

    std::vector<std::pair<uint8_t, uint8_t> > neighborIterator;

    uint16_t *allPushedIndX;
    uint16_t *allPushedIndY;

    uint16_t *queueIndX;
    uint16_t *queueIndY;

    visualization_msgs::MarkerArray costCubes;
    visualization_msgs::Marker costCube;

    std::vector<point_height> pc_height;
    // Model parameter for ground plane fitting
    // The ground plane model is: ax+by+cz+d=0
    // Here normal:=[a,b,c], d=d
    // th_dist_d_ = threshold_dist - d 
    float d_;
    Eigen::MatrixXf normal_;
    float th_dist_d_;

    std::vector<cv::Scalar> _colors;
    pcl::VoxelGrid<PointType> downSizeFilter;
    float _clustering_ranges[4] = {15.0, 30.0, 45.0, 60.0};
    float _clustering_distances[5] = {0.5, 1.1, 1.6, 2.1, 2.6};

    // Eigen::Affine3f Ext_Livox = Eigen::Affine3f::Identity();
public:
    ImageProjection():
        nh("~"){
        // 订阅来自velodyne雷达驱动的topic ("/velodyne_points")
        subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>("/livox_pcl0", 1, &ImageProjection::cloudHandler, this);
        
        pubFullCloud = nh.advertise<sensor_msgs::PointCloud2> ("/full_cloud_projected", 1);
        pubFullInfoCloud = nh.advertise<sensor_msgs::PointCloud2> ("/full_cloud_info", 1);

        pubGroundCloud = nh.advertise<sensor_msgs::PointCloud2> ("/ground_cloud", 1);
        pubSegmentedCloud = nh.advertise<sensor_msgs::PointCloud2> ("/segmented_cloud", 1);
        pubSegmentedCloudPure = nh.advertise<sensor_msgs::PointCloud2> ("/segmented_cloud_pure", 1);
        pubSegmentedCloudInfo = nh.advertise<cloud_msgs::cloud_info> ("/segmented_cloud_info", 1);
        // pubOutlierCloud = nh.advertise<sensor_msgs::PointCloud2> ("/outlier_cloud", 1);
        pubClusterCloud = nh.advertise<sensor_msgs::PointCloud2> ("/cluster_cloud", 1);

        pubMarkerArray = nh.advertise<visualization_msgs::MarkerArray>("/TEXT_VIEW_ARRAY", 10);

        pubPlaneEquation = nh.advertise<std_msgs::Float64MultiArray>("/plane_equation", 1);

        nanPoint.x = std::numeric_limits<float>::quiet_NaN();
        nanPoint.y = std::numeric_limits<float>::quiet_NaN();
        nanPoint.z = std::numeric_limits<float>::quiet_NaN();
        nanPoint.intensity = -1;

        allocateMemory();
        resetParameters();
    }

    // 析构函数
    ~ImageProjection(){}

    // 初始化各类参数以及分配内存
    void allocateMemory(){

        // 清空各类点云容器
        laserCloudIn.reset(new pcl::PointCloud<PointType>());

        fullCloud.reset(new pcl::PointCloud<PointType>());
        fullInfoCloud.reset(new pcl::PointCloud<PointType>());

        groundCloud.reset(new pcl::PointCloud<PointType>());
        segmentedCloud.reset(new pcl::PointCloud<PointType>());
        segmentedCloudPure.reset(new pcl::PointCloud<PointType>());
        // outlierCloud.reset(new pcl::PointCloud<PointType>());
        groundCloudSeeds.reset(new pcl::PointCloud<PointType>());

        colored_clustered_cloud_ptr.reset(new pcl::PointCloud<pcl::PointXYZRGB>());

        fullCloud->points.resize(N_SCAN*Horizon_SCAN);
        fullInfoCloud->points.resize(N_SCAN*Horizon_SCAN);

        segMsg.startRingIndex.assign(N_SCAN, 0);
        segMsg.endRingIndex.assign(N_SCAN, 0);

        segMsg.segmentedCloudGroundFlag.assign(N_SCAN*Horizon_SCAN, false);
        segMsg.segmentedCloudColInd.assign(N_SCAN*Horizon_SCAN, 0);
        segMsg.segmentedCloudRange.assign(N_SCAN*Horizon_SCAN, 0);

		// labelComponents函数中用到了这个矩阵
		// 该矩阵用于求某个点的上下左右4个邻接点
        std::pair<int8_t, int8_t> neighbor;
        neighbor.first = -1; neighbor.second =  0; neighborIterator.push_back(neighbor);
        neighbor.first =  0; neighbor.second =  1; neighborIterator.push_back(neighbor);
        neighbor.first =  0; neighbor.second = -1; neighborIterator.push_back(neighbor);
        neighbor.first =  1; neighbor.second =  0; neighborIterator.push_back(neighbor);

        allPushedIndX = new uint16_t[N_SCAN*Horizon_SCAN];
        allPushedIndY = new uint16_t[N_SCAN*Horizon_SCAN];

        queueIndX = new uint16_t[N_SCAN*Horizon_SCAN];
        queueIndY = new uint16_t[N_SCAN*Horizon_SCAN];

        #ifdef IS_CLUSTERS
        tree.reset(new pcl::search::KdTree<PointType>);
        // tree.reset(new pcl::KdTreeFLANN<PointType>);
        ec.setClusterTolerance(clusterTolerance);//设置近邻搜索半径
        ec.setMinClusterSize(minsize);//设置一个类需要的最小的点数
        ec.setMaxClusterSize(maxsize);//设置一个类需要的最大的点数
        #endif

        pc_height.resize(N_SCAN*Horizon_SCAN);

        // Eigen::Vector3f Ext_trans(ext_livox[0], ext_livox[1], ext_livox[2]);
        // Eigen::AngleAxisf rollAngle(ext_livox[3], Eigen::Vector3f::UnitX());
        // Eigen::AngleAxisf pitchAngle(ext_livox[4], Eigen::Vector3f::UnitY());
        // Eigen::AngleAxisf yawAngle(ext_livox[5], Eigen::Vector3f::UnitZ()); 
        // Eigen::Quaternionf quaternion;
        // quaternion=yawAngle*pitchAngle*rollAngle;
        // Ext_Livox.pretranslate(Ext_trans);
        // Ext_Livox.rotate(quaternion);

        // generateColors(_colors, 255);
        // 下采样滤波器设置叶子间距，就是格子之间的最小距离
        downSizeFilter.setLeafSize(0.4, 0.4, 0.4);
    }

	// 初始化/重置各类参数内容
    void resetParameters(){
        laserCloudIn->clear();
        groundCloud->clear();
        segmentedCloud->clear();
        segmentedCloudPure->clear();
        // outlierCloud->clear();
        groundCloudSeeds->clear();
        colored_clustered_cloud_ptr->clear();

        rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(0));
        groundMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_8S, cv::Scalar::all(0));
        labelMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32S, cv::Scalar::all(0));
        labelCount = 1;

        std::fill(fullCloud->points.begin(), fullCloud->points.end(), nanPoint);
        std::fill(fullInfoCloud->points.begin(), fullInfoCloud->points.end(), nanPoint);
    }

    void cloudHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg){
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        // msg->pcl
        copyPointCloud(laserCloudMsg);
        // findStartEndAngle();
        // projectPointCloudtoBase_Link();
        ransac_groundRemoval();
        // groundRemoval();
        cloudSegmentation();
        publishCloud();
        resetParameters();
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
        // std::cout << "solve time cost = " << time_used.count() << " seconds. " << std::endl;
        ROS_DEBUG("solve time cost = %f seconds.", time_used.count());
    }

    static void downsamplePoints(const cv::Mat& src, cv::Mat& dst, size_t count)
    {
        CV_Assert(count >= 2);
        CV_Assert(src.cols == 1 || src.rows == 1);
        CV_Assert(src.total() >= count);
        CV_Assert(src.type() == CV_8UC3);

        dst.create(1, (int)count, CV_8UC3);
        // TODO: optimize by exploiting symmetry in the distance matrix
        cv::Mat dists((int)src.total(), (int)src.total(), CV_32FC1, cv::Scalar(0));
        if (dists.empty())
            std::cerr << "Such big matrix cann't be created." << std::endl;

        for (int i = 0; i < dists.rows; i++)
        {
            for (int j = i; j < dists.cols; j++)
            {
            float dist = (float)cv::norm(src.at<cv::Point3_<uchar> >(i) - src.at<cv::Point3_<uchar> >(j));
            dists.at<float>(j, i) = dists.at<float>(i, j) = dist;
            }
        }

        double maxVal;
        cv::Point maxLoc;
        minMaxLoc(dists, 0, &maxVal, 0, &maxLoc);

        dst.at<cv::Point3_<uchar> >(0) = src.at<cv::Point3_<uchar> >(maxLoc.x);
        dst.at<cv::Point3_<uchar> >(1) = src.at<cv::Point3_<uchar> >(maxLoc.y);

        cv::Mat activedDists(0, dists.cols, dists.type());
        cv::Mat candidatePointsMask(1, dists.cols, CV_8UC1, cv::Scalar(255));
        activedDists.push_back(dists.row(maxLoc.y));
        candidatePointsMask.at<uchar>(0, maxLoc.y) = 0;

        for (size_t i = 2; i < count; i++)
        {
            activedDists.push_back(dists.row(maxLoc.x));
            candidatePointsMask.at<uchar>(0, maxLoc.x) = 0;

            cv::Mat minDists;
            reduce(activedDists, minDists, 0, CV_REDUCE_MIN);
            minMaxLoc(minDists, 0, &maxVal, 0, &maxLoc, candidatePointsMask);
            dst.at<cv::Point3_<uchar> >((int)i) = src.at<cv::Point3_<uchar> >(maxLoc.x);
        }
    }

    void generateColors(std::vector<cv::Scalar>& colors, size_t count, size_t factor = 100)
    {
        if (count < 1)
            return;

        colors.resize(count);

        if (count == 1)
        {
            colors[0] = cv::Scalar(0, 0, 255);  // red
            return;
        }
        if (count == 2)
        {
            colors[0] = cv::Scalar(0, 0, 255);  // red
            colors[1] = cv::Scalar(0, 255, 0);  // green
            return;
        }

        // Generate a set of colors in RGB space. A size of the set is severel times (=factor) larger then
        // the needed count of colors.
        cv::Mat bgr(1, (int)(count * factor), CV_8UC3);
        cv::randu(bgr, 0, 256);

        // Convert the colors set to Lab space.
        // Distances between colors in this space correspond a human perception.
        cv::Mat lab;
        cvtColor(bgr, lab, cv::COLOR_BGR2Lab);

        // Subsample colors from the generated set so that
        // to maximize the minimum distances between each other.
        // Douglas-Peucker algorithm is used for this.
        cv::Mat lab_subset;
        downsamplePoints(lab, lab_subset, count);

        // Convert subsampled colors back to RGB
        cv::Mat bgr_subset;
        cvtColor(lab_subset, bgr_subset, cv::COLOR_BGR2Lab);

        CV_Assert(bgr_subset.total() == count);
        for (size_t i = 0; i < count; i++)
        {
            cv::Point3_<uchar> c = bgr_subset.at<cv::Point3_<uchar> >((int)i);
            colors[i] = cv::Scalar(c.x, c.y, c.z);
        }
    }

    void copyPointCloud(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg){
        // 将ROS中的sensor_msgs::PointCloud2ConstPtr类型转换到pcl点云库指针
        cloudHeader = laserCloudMsg->header;
        pcl::fromROSMsg(*laserCloudMsg, *laserCloudIn);
        // std::cout<<"laserCloudIn size : "<<laserCloudIn->size()<<std::endl;
    }

    void projectPointCloudtoBase_Link(){
        // pcl::transformPointCloud(*laserCloudIn, *fullCloud, Ext_Livox);
        *fullCloud = *laserCloudIn;
    }

    // 移除地面点
    void groundRemoval(){
        size_t lowerInd, upperInd;
        float diffX, diffY, diffZ, angle;

        for(size_t i=0; i<Horizon_SCAN; i++)
        {
            for(size_t j=0; j<N_SCAN-1; j++)
            {
                lowerInd = i*N_SCAN+j;
                upperInd = lowerInd+1;

                // std::cout<<"line: "<<currPoint.intensity<<", ";
                // 排除仰天点
                if(laserCloudIn->points[lowerInd].z < 0.0)
                {
                    float rangeXY = laserCloudIn->points[lowerInd].x*laserCloudIn->points[lowerInd].x +
                                     laserCloudIn->points[lowerInd].y*laserCloudIn->points[lowerInd].y;
                    angle = atan2(std::fabs(laserCloudIn->points[lowerInd].z), sqrt(rangeXY)) * 180.0 / M_PI;

                    // std::cout<<"angle : "<<angle<<", ";
                    // 排除俯角太小的点，这些大概率不是打在地面
                    if(angle < 0.0) break;
                    // 剩下的就可能打在地面了

                    // 无效点
                    if(laserCloudIn->points[lowerInd].curvature ==0.0 ||
                        laserCloudIn->points[upperInd].curvature ==0.0)
                    {
                        groundMat.at<int8_t>(j, i) = -1;
                        continue;
                    }

                    // 由上下两线之间点的XYZ位置得到两线之间的俯仰角
                    // 如果俯仰角在10度以内，则判定(i,j)为地面点,groundMat[i][j]=1
                    // 否则，则不是地面点，进行后续操作
                    diffX = laserCloudIn->points[upperInd].x - laserCloudIn->points[lowerInd].x;
                    diffY = laserCloudIn->points[upperInd].y - laserCloudIn->points[lowerInd].y;
                    diffZ = laserCloudIn->points[upperInd].z - laserCloudIn->points[lowerInd].z;

                    angle = atan2(diffZ, sqrt(diffX*diffX + diffY*diffY) ) * 180 / M_PI;

                    // 标记为地面点
                    if (abs(angle - sensorMountAngle) <= 15){
                        groundMat.at<int8_t>(j, i) = 1;
                        groundMat.at<int8_t>(j+1, i) = 1;
                    }
                }
                else break;
            }
            // std::cout<<std::endl;
        }

        // 找到所有点中的地面点或者距离为FLT_MAX(rangeMat的初始值)的点，并将他们标记为-1
		// rangeMat[i][j]==0，代表的含义是什么？ 无效点
        // 遍历每条线
        for(size_t i=0; i<Horizon_SCAN; i++)
        {
            for(size_t j=0; j<N_SCAN; j++){
                // 顺便把rangeMat填充了
                rangeMat.at<float>(j, i) = std::sqrt(laserCloudIn->points[i*N_SCAN+j].x*laserCloudIn->points[i*N_SCAN+j].x +
                                                    laserCloudIn->points[i*N_SCAN+j].y*laserCloudIn->points[i*N_SCAN+j].y +
                                                    laserCloudIn->points[i*N_SCAN+j].z*laserCloudIn->points[i*N_SCAN+j].z);

                laserCloudIn->points[i*N_SCAN+j].curvature += i; // 记录该点在所属线束的位置
                if (groundMat.at<int8_t>(j, i) == 1 || rangeMat.at<float>(j, i) == 0.0){
                    labelMat.at<int>(j, i) = -1;
                }
            }
        }

		// // 如果有节点订阅groundCloud，那么就需要把地面点发布出来
		// // 具体实现过程：把点放到groundCloud队列中去
        // if (pubGroundCloud.getNumSubscribers() != 0){
        //     for(size_t i=0; i<Horizon_SCAN; i++){
        //         for(size_t j=0; j<N_SCAN; j++){
        //             if (groundMat.at<int8_t>(j, i) == 1)
        //                 groundCloud->push_back(laserCloudIn->points[j + i*N_SCAN]);
        //         }
        //     }
        //     // std::cout<<"groundCloud size: "<<groundCloud->size()<<std::endl;
        // }
    }

    /*
    @brief The function to estimate plane model. The
    model parameter `normal_` and `d_`, and `th_dist_d_`
    is set here.
    The main step is performed SVD(UAV) on covariance matrix.
    Taking the sigular vector in U matrix according to the smallest
    sigular value in A, as the `normal_`. `d_` is then calculated 
    according to mean ground points.

    @param g_ground_pc:global ground pointcloud ptr.
    
*/
void estimate_plane_(void){
    // Create covarian matrix in single pass.
    // TODO: compare the efficiency.
    // 协方差矩阵
    Eigen::Matrix3f cov;
    Eigen::Vector4f pc_mean;
    // 计算地面种子的协方差和均值
    pcl::computeMeanAndCovarianceMatrix(*groundCloudSeeds, cov, pc_mean);
    // Singular Value Decomposition: SVD
    // 计算协方差矩阵的奇异值
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(cov,Eigen::DecompositionOptions::ComputeFullU);
    // use the least singular vector as normal
    // 挑选出最小的特征值对应的特征向量，这是地平面的法向量
    normal_ = (svd.matrixU().col(2));
    // mean ground seeds value
    // 地面种子均值，均值点必经过地平面
    Eigen::Vector3f seeds_mean = pc_mean.head<3>();

    // according to normal.T*[x,y,z] = -d
    // 计算空间平面方程的d
    d_ = -(normal_.transpose()*seeds_mean)(0,0);
    // set distance threhold to `th_dist - d`
    // 设置距离阈值
    th_dist_d_ = th_dist - d_;
 
    // return the equation parameters
}

    // 移除地面点
    void ransac_groundRemoval(){
        pc_height.resize(N_SCAN*Horizon_SCAN);

        for(size_t i=0; i<Horizon_SCAN; i++)
        {
            for(size_t j=0; j<N_SCAN; j++)
            {
                // 顺便把rangeMat填充了
                rangeMat.at<float>(j, i) = std::sqrt(laserCloudIn->points[i*N_SCAN+j].x*laserCloudIn->points[i*N_SCAN+j].x +
                                                    laserCloudIn->points[i*N_SCAN+j].y*laserCloudIn->points[i*N_SCAN+j].y +
                                                    laserCloudIn->points[i*N_SCAN+j].z*laserCloudIn->points[i*N_SCAN+j].z);
                laserCloudIn->points[i*N_SCAN+j].curvature += i; // 记录该点在所属线束的位置

                pc_height[i*N_SCAN+j].value = laserCloudIn->points[i*N_SCAN+j].z;
                pc_height[i*N_SCAN+j].ind = i*N_SCAN+j;
            }
        }

        // 按点的z轴高度升序重排列
        std::sort(pc_height.begin(), pc_height.end(), by_height());

        // 去除由于测量噪声造成的无效点，无效点特点是比地面还要低0.2个livox_height
        std::vector<point_height>::iterator it = pc_height.begin();
        for(int i=0; i<pc_height.size(); i++)
        {
            if(pc_height[i].value < -1.2*livox_height)
                it++;
            else
                break;
        }
        // 从begin开始删除容器内前it个元素
        pc_height.erase(pc_height.begin(), it);

        // 提取地面种子
        // xtract initial seeds of the given pointcloud sorted segment.
        // This function filter ground seeds points accoring to heigt.
        // LPR是一部分最低点的均值
        float sum = 0;
        int cnt = 0;

        // 计算若干个最低高度点均值
        for(int i=0; i<pc_height.size() && cnt<num_lpr; i++)
        {
            sum += pc_height[i].value;
            cnt++;
        }
        // 计算均值
        float lpr_height = cnt!=0?sum/cnt:0.0;
        groundCloudSeeds->clear();
        // 提取地面种子
        for(int i=0; i<pc_height.size(); i++)
        {
            if(pc_height[i].value < lpr_height + th_seeds /*&&
                (laserCloudIn->points[pc_height[i].ind].x*laserCloudIn->points[pc_height[i].ind].x+
                laserCloudIn->points[pc_height[i].ind].y*laserCloudIn->points[pc_height[i].ind].y<10.0)*/)
                groundCloudSeeds->push_back(laserCloudIn->points[pc_height[i].ind]);
        }

        for(int i=0; i<num_iter; i++)
        {
            // 求取地平面的法向量、地平面方程
            estimate_plane_();
            groundCloudSeeds->clear();

            // 将点云中的点(x,y,z)^T按顺序排列成一个矩阵
            Eigen::MatrixXf points(laserCloudIn->points.size(),3);
            int j = 0;
            for(auto p:laserCloudIn->points)
            {
                points.row(j++)<<p.x, p.y, p.z;
            }
            // 将所有点套入平面方程
            Eigen::VectorXf result = points*normal_;
            // 根据阈值对点进行标记
            for(int r=0; r<result.rows(); r++)
            {
                if(result[r]<th_dist_d_)
                {
                    // 标记地面点
                    groundCloudSeeds->points.push_back(laserCloudIn->points[r]);
                }
            }
        }

        std_msgs::Float64MultiArray planeMsg;
        planeMsg.data.push_back(cloudHeader.stamp.toSec());
        planeMsg.data.push_back(normal_(0, 0));
        planeMsg.data.push_back(normal_(1, 0));
        planeMsg.data.push_back(normal_(2, 0));
        planeMsg.data.push_back(d_);
        pubPlaneEquation.publish(planeMsg);

        // std::cout<<"normal_ : "<<normal_.transpose()<<", d_ : "<<d_<<std::endl;
        for(auto p:groundCloudSeeds->points)
        {
            groundCloud->push_back(p);
            // 标记地面点
            groundMat.at<int8_t>(int(p.intensity), int(p.curvature)) = 1;
        }
        // 找到所有点中的地面点或者距离为0(rangeMat的初始值)的点，并将他们标记为-1
		// rangeMat[i][j]==0，代表的含义是什么？ 无效点
        // 遍历每条线
        for(size_t i=0; i<Horizon_SCAN; i++)
        {
            for(size_t j=0; j<N_SCAN; j++){
                if (groundMat.at<int8_t>(j, i) == 1 || rangeMat.at<float>(j, i) == 0.0){
                    labelMat.at<int>(j, i) = -1;
                }
            }
        }
    }

    // 点云分块
    void cloudSegmentation(){
        // 这里逐条线进行遍历
        // for(size_t i=0; i<N_SCAN; i++){
        //     for(size_t j=0; j<Horizon_SCAN; j++){
        //         // 需要选择不是地面点(labelMat[i][j]!=-1)和没被舍弃的点
        //         // if (labelMat.at<int>(i, j) >= 0){
        //         if (labelMat.at<int>(i,j) >= 0){
        //             segmentedCloudPure->push_back(laserCloudIn->points[i + j*N_SCAN]);
        //             segmentedCloudPure->points.back().intensity = labelMat.at<int>(i, j);
        //         }
        //     }
        // }
        // // // labelComponents();
        // segmentByDistance(segmentedCloudPure, colored_clustered_cloud_ptr);

        int sizeOfSegCloud = 0;
        for (size_t i = 0; i < N_SCAN; ++i) {
			// segMsg.startRingIndex[i]
			// segMsg.endRingIndex[i]
			// 表示第i线的点云起始序列和终止序列
			// 以开始线后的第6线为开始，以结束线前的第6线为结束
            segMsg.startRingIndex[i] = sizeOfSegCloud-1 + 5;
            for (size_t j = 0; j < Horizon_SCAN; ++j) {
                // 找到可用的特征点或者地面点(不选择labelMat[i][j]=0的点)
                if (labelMat.at<int>(i,j) >= 0 || groundMat.at<int8_t>(i,j) == 1){
                    // labelMat数值为999999表示这个点是因为聚类数量不够30而被舍弃的点
					// 需要舍弃的点直接continue跳过本次循环，
					// 当列数为5的倍数，并且行数较大，可以认为非地面点的，将它保存进异常点云(界外点云)中
					// 然后再跳过本次循环
                    // if (labelMat.at<int>(i,j) == 999999){
                    //     if (i > groundScanInd && j % 5 == 0){
                    //         outlierCloud->push_back(laserCloudIn->points[i + j*N_SCAN]);
                    //         continue;
                    //     }else{
                    //         continue;
                    //     }
                    // }

                    // 如果是地面点,对于列数不为5的倍数的，直接跳过不处理
                    if (groundMat.at<int8_t>(i,j) == 1){
                        // std::cout<<"rangeMat.at<float>(i,j) = "<<rangeMat.at<float>(i,j)<<", ";
                        // if(rangeMat.at<float>(i,j) < 50.0)
                        //     continue;
                        if (j%5!=0 && j>5 && j<Horizon_SCAN-5)
                            continue;
                        float DisXY = std::sqrt(laserCloudIn->points[i + j*N_SCAN].x * laserCloudIn->points[i + j*N_SCAN].x +
                                                laserCloudIn->points[i + j*N_SCAN].y * laserCloudIn->points[i + j*N_SCAN].y);
                        if (j%10!=0 && j>5 && j<Horizon_SCAN-5 && DisXY<30.0)
                            continue;
                    }
					// 上面多个if语句已经去掉了不符合条件的点，这部分直接进行信息的拷贝和保存操作
					// 保存完毕后sizeOfSegCloud递增
                    segMsg.segmentedCloudGroundFlag[sizeOfSegCloud] = (groundMat.at<int8_t>(i,j) == 1); // 标记地面点
                    segMsg.segmentedCloudColInd[sizeOfSegCloud] = j; // 标记该点在所属线上的位置
                    segMsg.segmentedCloudRange[sizeOfSegCloud]  = rangeMat.at<float>(i,j); // 该点的探测深度
                    segmentedCloud->push_back(laserCloudIn->points[i + j*N_SCAN]);
                    ++sizeOfSegCloud;
                }
            }
            // 以结束线前的第5线为结束
            segMsg.endRingIndex[i] = sizeOfSegCloud-1 - 5;
        }
        // std::cout<<std::endl;
        // 如果有节点订阅SegmentedCloudPure,
		// 那么把点云数据保存到segmentedCloudPure中去
        if (pubSegmentedCloudPure.getNumSubscribers() != 0){
            // 这里逐条线进行遍历
            for(size_t i=0; i<N_SCAN; i++){
                for(size_t j=0; j<Horizon_SCAN; j++){
					// 需要选择不是地面点(labelMat[i][j]!=-1)和没被舍弃的点
                    // if (labelMat.at<int>(i, j) >= 0){
                    if (labelMat.at<int>(i,j) >= 0){
                        segmentedCloudPure->push_back(laserCloudIn->points[i + j*N_SCAN]);
                        segmentedCloudPure->points.back().intensity = labelMat.at<int>(i, j);
                    }
                }
            }
        }
    }

    void labelComponents(){
        #ifdef IS_CLUSTERS
        clusterIndices.clear();
        clusters.clear();

        tree->setInputCloud(segmentedCloudPure);
        ec.setSearchMethod(tree);//设置搜索方法
        ec.setInputCloud(segmentedCloudPure); // feed point cloud
        ec.extract(clusterIndices); // 得到所有类别的索引 clusterIndices  11类
        
        bool once = true;
        // 将得到的所有类的索引分别在点云中找到，即每一个索引形成了一个类
        for (pcl::PointIndices getIndices : clusterIndices)
        {
            pcl::PointCloud<PointType> cloudCluster;
            float x=0.0, y=0.0, z=0.0;
            //PtCdtr<PointT> cloudCluster(new pcl::PointCloud<PointT>);
            // For each point indice in each cluster
            for (int index : getIndices.indices) 
            {
                cloudCluster.push_back(segmentedCloudPure->points[index]);
                // x += segmentedCloudPure->points[index].x;
                // y += segmentedCloudPure->points[index].y;
                // z += segmentedCloudPure->points[index].z;
            }
            cloudCluster.width = cloudCluster.size();
            cloudCluster.height = 1;
            cloudCluster.is_dense = true;
            clusters.push_back(cloudCluster);

            // x = x/cloudCluster.size();
            // y = y/cloudCluster.size();
            // z = z/cloudCluster.size();

            // float sx=0.0, sy=0.0, sz=0.0;
            // for (int index : getIndices.indices)
            // {
            //     sx = std::max(sx, std::fabs(segmentedCloudPure->points[index].x-x));
            //     sy = std::max(sy, std::fabs(segmentedCloudPure->points[index].y-y));
            //     sz = std::max(sz, std::fabs(segmentedCloudPure->points[index].z-z));
            // }
            PointType minPoint, maxPoint;
            pcl::getMinMax3D(cloudCluster, minPoint, maxPoint);

            if (once) {
                costCube.action = 3;
                once = false;
            } else {
                costCube.action = 0;
            }
            costCube.header.frame_id = "/livox";
            costCube.header.stamp = cloudHeader.stamp;
            costCube.id = clusters.size()-1;
            costCube.type = visualization_msgs::Marker::CUBE;
            costCube.pose.position.x = maxPoint.x - (maxPoint.x - minPoint.x)/2;
            costCube.pose.position.y = maxPoint.y - (maxPoint.y - minPoint.y)/2;
            costCube.pose.position.z = maxPoint.z - (maxPoint.z - minPoint.z)/2;
            costCube.color.a = 0.5;
            costCube.color.r = 255;
            costCube.color.g = 0;
            costCube.color.b = 0;
            costCube.scale.x = (maxPoint.x - minPoint.x);
            costCube.scale.y = (maxPoint.y - minPoint.y);
            costCube.scale.z = (maxPoint.z - minPoint.z);
            costCubes.markers.push_back(costCube);
        }
        pubMarkerArray.publish(costCubes);
        // std::cout<<"clusters size : "<<clusters.size()<<std::endl;
        #endif
    }

    void checkClusterMerge(size_t in_cluster_id, std::vector<ClusterPtr> &in_clusters,
                        std::vector<bool> &in_out_visited_clusters, std::vector<size_t> &out_merge_indices,
                        double in_merge_threshold)
    {
    // std::cout << "checkClusterMerge" << std::endl;
    PointType point_a = in_clusters[in_cluster_id]->GetCentroid();
    for (size_t i = 0; i < in_clusters.size(); i++)
    {
        if (i != in_cluster_id && !in_out_visited_clusters[i])
        {
        PointType point_b = in_clusters[i]->GetCentroid();
        double distance = sqrt(pow(point_b.x - point_a.x, 2) + pow(point_b.y - point_a.y, 2));
        if (distance <= in_merge_threshold)
        {
            in_out_visited_clusters[i] = true;
            out_merge_indices.push_back(i);
            // std::cout << "Merging " << in_cluster_id << " with " << i << " dist:" << distance << std::endl;
            checkClusterMerge(i, in_clusters, in_out_visited_clusters, out_merge_indices, in_merge_threshold);
        }
        }
    }
    }

    void mergeClusters(const std::vector<ClusterPtr> &in_clusters, std::vector<ClusterPtr> &out_clusters,
                    std::vector<size_t> in_merge_indices, const size_t &current_index,
                    std::vector<bool> &in_out_merged_clusters)
    {
    // std::cout << "mergeClusters:" << in_merge_indices.size() << std::endl;
    pcl::PointCloud<pcl::PointXYZRGB> sum_cloud;
    pcl::PointCloud<PointType> mono_cloud;
    ClusterPtr merged_cluster(new Cluster());
    for (size_t i = 0; i < in_merge_indices.size(); i++)
    {
        sum_cloud += *(in_clusters[in_merge_indices[i]]->GetCloud());
        in_out_merged_clusters[in_merge_indices[i]] = true;
    }
    std::vector<int> indices(sum_cloud.points.size(), 0);
    for (size_t i = 0; i < sum_cloud.points.size(); i++)
    {
        indices[i] = i;
    }

    if (sum_cloud.points.size() > 0)
    {
        pcl::copyPointCloud(sum_cloud, mono_cloud);
        merged_cluster->SetCloud(mono_cloud.makeShared(), indices, cloudHeader, current_index,
                                (int) _colors[current_index].val[0], (int) _colors[current_index].val[1],
                                (int) _colors[current_index].val[2], "", false);
        out_clusters.push_back(merged_cluster);
    }
    }

    void checkAllForMerge(std::vector<ClusterPtr> &in_clusters, std::vector<ClusterPtr> &out_clusters,
                        float in_merge_threshold)
    {
    // std::cout << "checkAllForMerge" << std::endl;
    std::vector<bool> visited_clusters(in_clusters.size(), false);
    std::vector<bool> merged_clusters(in_clusters.size(), false);
    size_t current_index = 0;
    for (size_t i = 0; i < in_clusters.size(); i++)
    {
        if (!visited_clusters[i])
        {
        visited_clusters[i] = true;
        std::vector<size_t> merge_indices;
        checkClusterMerge(i, in_clusters, visited_clusters, merge_indices, in_merge_threshold);
        mergeClusters(in_clusters, out_clusters, merge_indices, current_index++, merged_clusters);
        }
    }
    for (size_t i = 0; i < in_clusters.size(); i++)
    {
        // check for clusters not merged, add them to the output
        if (!merged_clusters[i])
        {
        out_clusters.push_back(in_clusters[i]);
        }
    }

    // ClusterPtr cluster(new Cluster());
    }

    std::vector<ClusterPtr> clusterAndColor(const pcl::PointCloud<PointType>::Ptr in_cloud_ptr,
                                        std::vector<ClusterPtr> &clusters,
                                        double in_max_cluster_distance = 0.5)
    {
        pcl::search::KdTree<PointType>::Ptr tree(new pcl::search::KdTree<PointType>);
        // std::vector<pcl::PointCloud<PointType>> clusters;//保存分割后的所有类 每一类为一个点云
        clusters.clear();
        // // 欧式聚类对检测到的障碍物进行分组
        int minsize = 50;
        int maxsize = 6000;
        std::vector<pcl::PointIndices> clusterIndices;// 创建索引类型对象
        pcl::EuclideanClusterExtraction<PointType> ec; // 欧式聚类对象

        // create 2d pc
        pcl::PointCloud<PointType>::Ptr cloud_2d(new pcl::PointCloud<PointType>);
        pcl::copyPointCloud(*in_cloud_ptr, *cloud_2d);
        // make it flat
        for (size_t i = 0; i < cloud_2d->points.size(); i++)
        {
            cloud_2d->points[i].z = 0;
        }

        if (cloud_2d->points.size() > 0)
            tree->setInputCloud(cloud_2d);

        ec.setClusterTolerance(in_max_cluster_distance);  //
        ec.setMinClusterSize(minsize);
        ec.setMaxClusterSize(maxsize);
        ec.setSearchMethod(tree);
        ec.setInputCloud(cloud_2d);
        ec.extract(clusterIndices);
        // use indices on 3d cloud

        /////////////////////////////////
        //---	3. Color clustered points
        /////////////////////////////////
        unsigned int k = 0;
        for (pcl::PointIndices getIndices : clusterIndices)
        {
            ClusterPtr cluster(new Cluster());
            cluster->SetCloud(in_cloud_ptr, getIndices.indices, cloudHeader, k, (int) _colors[k].val[0],
                            (int) _colors[k].val[1],
                            (int) _colors[k].val[2], "", false);
            clusters.push_back(cluster);
            k++;
        }
        return clusters;
    }

    void segmentByDistance(const pcl::PointCloud<PointType>::Ptr in_cloud, 
                            pcl::PointCloud<pcl::PointXYZRGB>::Ptr out_cloud,
                            double in_max_cluster_distance = 0.75)
    {
        pcl::PointCloud<PointType>::Ptr in_cloudDS(new pcl::PointCloud<PointType>());
        downSizeFilter.setInputCloud(in_cloud);
        downSizeFilter.filter(*in_cloudDS);
        std::vector<ClusterPtr> clusters;

        bool _use_multiple_thres = false;
        if(!_use_multiple_thres)
        {
            clusterAndColor(in_cloudDS, clusters, in_max_cluster_distance);
        }
        else
        {
            std::vector<pcl::PointCloud<PointType>::Ptr> cloud_segments_array(5);
            for (unsigned int i = 0; i < cloud_segments_array.size(); i++)
            {
                pcl::PointCloud<PointType>::Ptr tmp_cloud(new pcl::PointCloud<PointType>);
                cloud_segments_array[i] = tmp_cloud;
            }

            for (unsigned int i = 0; i < in_cloudDS->points.size(); i++)
            {
                PointType current_point;
                current_point.x = in_cloudDS->points[i].x;
                current_point.y = in_cloudDS->points[i].y;
                current_point.z = in_cloudDS->points[i].z;

                float origin_distance = sqrt(pow(current_point.x, 2) + pow(current_point.y, 2));

                if (origin_distance < _clustering_ranges[0])
                {
                    cloud_segments_array[0]->points.push_back(current_point);
                }
                else if (origin_distance < _clustering_ranges[1])
                {
                    cloud_segments_array[1]->points.push_back(current_point);

                }else if (origin_distance < _clustering_ranges[2])
                {
                    cloud_segments_array[2]->points.push_back(current_point);

                }else if (origin_distance < _clustering_ranges[3])
                {
                    cloud_segments_array[3]->points.push_back(current_point);

                }else
                {
                    cloud_segments_array[4]->points.push_back(current_point);
                }
            }
            std::vector<ClusterPtr> local_clusters;
            for (unsigned int i = 0; i < cloud_segments_array.size(); i++)
            {
                clusterAndColor(cloud_segments_array[i], local_clusters, _clustering_distances[i]);
                clusters.insert(clusters.end(), local_clusters.begin(), local_clusters.end());
            }
        }

        // Clusters can be merged or checked in here
        //....
        // check for mergable clusters
        // std::vector<ClusterPtr> mid_clusters;
        // std::vector<ClusterPtr> final_clusters;

        // if (clusters.size() > 0)
        //     checkAllForMerge(clusters, mid_clusters, 1.5);
        // else
        //     mid_clusters = clusters;

        // if (mid_clusters.size() > 0)
        //     checkAllForMerge(mid_clusters, final_clusters, 1.5);
        // else
        //     final_clusters = mid_clusters;

        for(int i=0; i<clusters.size(); i++)
        {
            *colored_clustered_cloud_ptr += *(clusters[i]->GetCloud());
        }
    }

    // 发布各类点云内容
    void publishCloud(){
        // 发布cloud_msgs::cloud_info消息
        segMsg.header = cloudHeader;
        pubSegmentedCloudInfo.publish(segMsg);

        sensor_msgs::PointCloud2 laserCloudTemp;

		// pubOutlierCloud发布界外点云
        // pcl::toROSMsg(*outlierCloud, laserCloudTemp);
        // laserCloudTemp.header.stamp = cloudHeader.stamp;
        // laserCloudTemp.header.frame_id = "livox";
        // pubOutlierCloud.publish(laserCloudTemp);

		// pubSegmentedCloud发布分块点云
        pcl::toROSMsg(*segmentedCloud, laserCloudTemp);
        laserCloudTemp.header.stamp = cloudHeader.stamp;
        laserCloudTemp.header.frame_id = "/livox";
        pubSegmentedCloud.publish(laserCloudTemp);

        if (pubFullCloud.getNumSubscribers() != 0){
            pcl::toROSMsg(*fullCloud, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = "/livox";
            pubFullCloud.publish(laserCloudTemp);
        }

        if (pubGroundCloud.getNumSubscribers() != 0){
            pcl::toROSMsg(*groundCloud, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = "/livox";
            pubGroundCloud.publish(laserCloudTemp);
        }

        if (pubSegmentedCloudPure.getNumSubscribers() != 0){
            pcl::toROSMsg(*segmentedCloudPure, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = "/livox";
            pubSegmentedCloudPure.publish(laserCloudTemp);
        }

        if (pubFullInfoCloud.getNumSubscribers() != 0){
            pcl::toROSMsg(*fullInfoCloud, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = "/livox";
            pubFullInfoCloud.publish(laserCloudTemp);
        }

        if(pubClusterCloud.getNumSubscribers() != 0)
        {
            pcl::toROSMsg(*colored_clustered_cloud_ptr, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = "/livox";
            pubClusterCloud.publish(laserCloudTemp);
        }
    }
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "image_projection_node");

  ImageProjection featureHandler;

  ROS_INFO("\033[1;32m---->\033[0m Feature Extraction Module Started.");

  ros::spin();
  return 0;
}