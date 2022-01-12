#include <unordered_map>
#include "livox_ros_driver/CustomMsg.h"
#include "lins_livox/common.h"

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>

#include <gtsam/nonlinear/ISAM2.h>

using namespace gtsam;

class mapOptimization{
private:
    NonlinearFactorGraph gtSAMgraph;
    Values initialEstimate;
    Values optimizedEstimate;
    ISAM2 *isam;
    Values isamCurrentEstimate;

    noiseModel::Diagonal::shared_ptr priorNoise;
    noiseModel::Diagonal::shared_ptr odometryNoise;
    noiseModel::Diagonal::shared_ptr constraintNoise;

    ros::NodeHandle nh;

    ros::Publisher pubLaserCloudSurround;
    ros::Publisher pubOdomAftMapped;
    ros::Publisher pubKeyPoses;

    ros::Publisher pubHistoryKeyFrames;
    ros::Publisher pubIcpKeyFrames;
    ros::Publisher pubRecentKeyFrames;

    ros::Subscriber subLaserCloudCornerLast;
    ros::Subscriber subLaserCloudSurfLast;
    ros::Subscriber subLaserOdometry;

    nav_msgs::Odometry odomAftMapped;
    tf::StampedTransform aftMappedTrans;
    tf::TransformBroadcaster tfBroadcaster;

    tf::StampedTransform aftMappedOdomTrans;

    std::vector<pcl::PointCloud<PointType>::Ptr> cornerCloudKeyFrames;
    std::vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames;

    std::deque<pcl::PointCloud<PointType>::Ptr> recentCornerCloudKeyFrames;
    std::deque<pcl::PointCloud<PointType>::Ptr> recentSurfCloudKeyFrames;
    // deque<pcl::PointCloud<PointType>::Ptr> recentOutlierCloudKeyFrames;
    int latestFrameID;

    std::vector<int> surroundingExistingKeyPosesID;
    std::deque<pcl::PointCloud<PointType>::Ptr> surroundingCornerCloudKeyFrames;
    std::deque<pcl::PointCloud<PointType>::Ptr> surroundingSurfCloudKeyFrames;
    // deque<pcl::PointCloud<PointType>::Ptr> surroundingOutlierCloudKeyFrames;
    
    PointType previousRobotPosPoint;
    PointType currentRobotPosPoint;
    Eigen::Quaternionf previousRobotAttPoint;
    Eigen::Quaternionf currentRobotAttPoint;

    // PointType(pcl::PointXYZI)的XYZI分别保存3个方向上的平移和一个索引(cloudKeyPoses3D->points.size())
    pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D;

    //PointTypePose的XYZI保存和cloudKeyPoses3D一样的内容，另外还保存RPY角度以及一个时间值timeLaserOdometry
    // pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D;
    pcl::PointCloud<myPointTypePose>::Ptr cloudKeyPoses6D;
    
    // 结尾有DS代表是downsize,进行过下采样
    pcl::PointCloud<PointType>::Ptr surroundingKeyPoses;
    pcl::PointCloud<PointType>::Ptr surroundingKeyPosesDS;

    pcl::PointCloud<PointType>::Ptr laserCloudCornerLast;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLast;
    pcl::PointCloud<PointType>::Ptr laserCloudCornerLastDS;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLastDS;

    // pcl::PointCloud<PointType>::Ptr laserCloudOutlierLast;
    // pcl::PointCloud<PointType>::Ptr laserCloudOutlierLastDS;

    pcl::PointCloud<PointType>::Ptr laserCloudSurfTotalLast;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfTotalLastDS;

    pcl::PointCloud<PointType>::Ptr laserCloudOri;
    pcl::PointCloud<PointType>::Ptr coeffSel;

    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap;
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMapDS;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMapDS;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurroundingKeyPoses;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses;

    
    pcl::PointCloud<PointType>::Ptr nearHistoryCornerKeyFrameCloud;
    pcl::PointCloud<PointType>::Ptr nearHistoryCornerKeyFrameCloudDS;
    pcl::PointCloud<PointType>::Ptr nearHistorySurfKeyFrameCloud;
    pcl::PointCloud<PointType>::Ptr nearHistorySurfKeyFrameCloudDS;

    pcl::PointCloud<PointType>::Ptr latestCornerKeyFrameCloud;
    pcl::PointCloud<PointType>::Ptr latestSurfKeyFrameCloud;
    pcl::PointCloud<PointType>::Ptr latestSurfKeyFrameCloudDS;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeGlobalMap;
    pcl::PointCloud<PointType>::Ptr globalMapKeyPoses;
    pcl::PointCloud<PointType>::Ptr globalMapKeyPosesDS;
    pcl::PointCloud<PointType>::Ptr globalMapKeyFrames;
    pcl::PointCloud<PointType>::Ptr globalMapKeyFramesDS;

    std::vector<int> pointSearchInd;
    std::vector<float> pointSearchSqDis;

    pcl::VoxelGrid<PointType> downSizeFilterCorner;
    pcl::VoxelGrid<PointType> downSizeFilterSurf;
    // pcl::VoxelGrid<PointType> downSizeFilterOutlier;
    pcl::VoxelGrid<PointType> downSizeFilterHistoryKeyFrames;
    pcl::VoxelGrid<PointType> downSizeFilterSurroundingKeyPoses;
    pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyPoses;
    pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyFrames;

    double timeLaserCloudCornerLast;
    double timeLaserCloudSurfLast;
    double timeLaserOdometry;
    double timeLaserCloudOutlierLast;
    double timeLastGloalMapPublish;

    bool newLaserCloudCornerLast;
    bool newLaserCloudSurfLast;
    bool newLaserOdometry;
    bool newLaserCloudOutlierLast;

    float transformLast[6];         // 保存上一次LM优化后的位姿
    Eigen::Quaternionf q_transformLast;
    Eigen::Vector3f v_transformLast;

    /*************高频转换量**************/
    // odometry计算得到的到世界坐标系下的转移矩阵
    float transformSum[6];
    Eigen::Quaternionf q_transformSum;
    Eigen::Vector3f v_transformSum;
    // 转移增量，只使用了后三个平移增量
    float transformIncre[6];
    Eigen::Quaternionf q_transformIncre;
    Eigen::Vector3f v_transformIncre;

    /*************低频转换量*************/
    // 以起始位置为原点的世界坐标系下的转换矩阵（猜测与调整的对象）
    // 1.里程计数据到来，更新这个预估值
    // 2.LM优化的时候会调整+dx
    // 3.LM优化后，与IMU有加权融合
    // 4.gtsam优化后，直接取优化值
    float transformTobeMapped[6];
    Eigen::Quaternionf q_transformTobeMapped;
    Eigen::Vector3f v_transformTobeMapped;
    // 存放mapping之前的Odometry计算的世界坐标系的转换矩阵（注：低频量，不一定与transformSum一样）
    float transformBefMapped[6];
    Eigen::Quaternionf q_transformBefMapped;
    Eigen::Vector3f v_transformBefMapped;
    // 存放mapping之后的经过mapping微调之后的转换矩阵
    // 1.LM优化后，会取融合后的transformTobeMapped[]
    // 2.gtsam优化后，直接取优化值
    float transformAftMapped[6];
    Eigen::Quaternionf q_transformAftMapped;
    Eigen::Vector3f v_transformAftMapped;

    // int imuPointerFront;
    // int imuPointerLast;

    // double imuTime[imuQueLength];
    // float imuRoll[imuQueLength];
    // float imuPitch[imuQueLength];

    std::mutex mtx;

    double timeLastProcessing;

    PointType pointOri, pointSel, pointProj, coeff;

    cv::Mat matA0;
    cv::Mat matB0;
    cv::Mat matX0;

    cv::Mat matA1;
    cv::Mat matD1;
    cv::Mat matV1;

    bool isDegenerate;
    Eigen::Matrix<float, 6, 6> matP;

    int laserCloudCornerFromMapDSNum;
    int laserCloudSurfFromMapDSNum;
    int laserCloudCornerLastDSNum;
    int laserCloudSurfLastDSNum;
    // int laserCloudOutlierLastDSNum;
    int laserCloudSurfTotalLastDSNum;

    bool potentialLoopFlag;
    double timeSaveFirstCurrentScanForLoopClosure;
    int closestHistoryFrameID;
    int latestFrameIDLoopCloure;

    bool aLoopIsClosed;

    float cRoll, sRoll, cPitch, sPitch, cYaw, sYaw, tX, tY, tZ;
    float ctRoll, stRoll, ctPitch, stPitch, ctYaw, stYaw, tInX, tInY, tInZ;

public:
mapOptimization():
        nh("~")
    {
        // 用于闭环图优化的参数设置，使用gtsam库
    	ISAM2Params parameters;
		parameters.relinearizeThreshold = 0.01;
		parameters.relinearizeSkip = 1;
        parameters.factorization = ISAM2Params::QR;
    	isam = new ISAM2(parameters);

        // 发布
        pubKeyPoses = nh.advertise<sensor_msgs::PointCloud2>("/key_pose_origin", 2);
        pubLaserCloudSurround = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surround", 2);
        pubOdomAftMapped = nh.advertise<nav_msgs::Odometry> ("/aft_mapped_to_init", 5);

        // 订阅，相机坐标系下的点
        subLaserCloudCornerLast = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_corner_last", 2, &mapOptimization::laserCloudCornerLastHandler, this);
        subLaserCloudSurfLast = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 2, &mapOptimization::laserCloudSurfLastHandler, this);
        // 相机导航坐标系n'的里程计
        subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("/laser_odom_to_init", 5, &mapOptimization::laserOdometryHandler, this);

        // 发布的
        pubHistoryKeyFrames = nh.advertise<sensor_msgs::PointCloud2>("/history_cloud", 2);
        pubIcpKeyFrames = nh.advertise<sensor_msgs::PointCloud2>("/corrected_cloud", 2);
        pubRecentKeyFrames = nh.advertise<sensor_msgs::PointCloud2>("/recent_cloud", 2);

        // 设置滤波时创建的体素大小为0.2m/0.4m立方体,下面的单位为m
        downSizeFilterCorner.setLeafSize(0.2, 0.2, 0.2);
        downSizeFilterSurf.setLeafSize(0.4, 0.4, 0.4);

        downSizeFilterHistoryKeyFrames.setLeafSize(0.4, 0.4, 0.4);
        downSizeFilterSurroundingKeyPoses.setLeafSize(1.0, 1.0, 1.0);

        downSizeFilterGlobalMapKeyPoses.setLeafSize(1.0, 1.0, 1.0);
        downSizeFilterGlobalMapKeyFrames.setLeafSize(0.4, 0.4, 0.4);

        odomAftMapped.header.frame_id = "/camera_init";
        odomAftMapped.child_frame_id = "/aft_mapped";
        // odomAftMapped.child_frame_id = "/livox";

        aftMappedTrans.frame_id_ = "/camera_init";
        aftMappedTrans.child_frame_id_ = "/aft_mapped";
        // aftMappedTrans.child_frame_id_ = "/livox";

        aftMappedOdomTrans.frame_id_ = "/camera_init";
        // aftMappedTrans.child_frame_id_ = "/aft_mapped";
        aftMappedOdomTrans.child_frame_id_ = "/livox";

        allocateMemory();
    }

    void allocateMemory(){

        cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        cloudKeyPoses6D.reset(new pcl::PointCloud<myPointTypePose>());

        kdtreeSurroundingKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());

        surroundingKeyPoses.reset(new pcl::PointCloud<PointType>());
        surroundingKeyPosesDS.reset(new pcl::PointCloud<PointType>());        

        laserCloudCornerLast.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfLast.reset(new pcl::PointCloud<PointType>());
        laserCloudCornerLastDS.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfLastDS.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfTotalLast.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfTotalLastDS.reset(new pcl::PointCloud<PointType>());

        laserCloudOri.reset(new pcl::PointCloud<PointType>());
        coeffSel.reset(new pcl::PointCloud<PointType>());

        laserCloudCornerFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudCornerFromMapDS.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMapDS.reset(new pcl::PointCloud<PointType>());

        kdtreeCornerFromMap.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeSurfFromMap.reset(new pcl::KdTreeFLANN<PointType>());

        
        nearHistoryCornerKeyFrameCloud.reset(new pcl::PointCloud<PointType>());
        nearHistoryCornerKeyFrameCloudDS.reset(new pcl::PointCloud<PointType>());
        nearHistorySurfKeyFrameCloud.reset(new pcl::PointCloud<PointType>());
        nearHistorySurfKeyFrameCloudDS.reset(new pcl::PointCloud<PointType>());

        latestCornerKeyFrameCloud.reset(new pcl::PointCloud<PointType>());
        latestSurfKeyFrameCloud.reset(new pcl::PointCloud<PointType>());
        latestSurfKeyFrameCloudDS.reset(new pcl::PointCloud<PointType>());

        kdtreeGlobalMap.reset(new pcl::KdTreeFLANN<PointType>());
        globalMapKeyPoses.reset(new pcl::PointCloud<PointType>());
        globalMapKeyPosesDS.reset(new pcl::PointCloud<PointType>());
        globalMapKeyFrames.reset(new pcl::PointCloud<PointType>());
        globalMapKeyFramesDS.reset(new pcl::PointCloud<PointType>());

        timeLaserCloudCornerLast = 0;
        timeLaserCloudSurfLast = 0;
        timeLaserOdometry = 0;
        timeLaserCloudOutlierLast = 0;
        timeLastGloalMapPublish = 0;

        timeLastProcessing = -1;

        newLaserCloudCornerLast = false;
        newLaserCloudSurfLast = false;

        newLaserOdometry = false;
        newLaserCloudOutlierLast = false;

        for (int i = 0; i < 6; ++i){
            transformLast[i] = 0;
            transformSum[i] = 0;
            transformIncre[i] = 0;
            transformTobeMapped[i] = 0;
            transformBefMapped[i] = 0;
            transformAftMapped[i] = 0;
        }

        q_transformSum = Eigen::Quaternionf::Identity();
        v_transformSum = Eigen::Vector3f::Zero();
        q_transformIncre = Eigen::Quaternionf::Identity();
        v_transformIncre = Eigen::Vector3f::Zero();
        q_transformTobeMapped = Eigen::Quaternionf::Identity();
        v_transformTobeMapped = Eigen::Vector3f::Zero();
        q_transformBefMapped = Eigen::Quaternionf::Identity();
        v_transformBefMapped = Eigen::Vector3f::Zero();
        q_transformAftMapped = Eigen::Quaternionf::Identity();
        v_transformAftMapped = Eigen::Vector3f::Zero();
        q_transformLast = Eigen::Quaternionf::Identity();
        v_transformLast = Eigen::Vector3f::Zero();

        // 初始化　先验噪声＼里程计噪声
        gtsam::Vector Vector6(6);
        gtsam::Vector OdometryVector6(6);
        Vector6 << 1e-6, 1e-6, 1e-6, 1e-8, 1e-8, 1e-6;
        OdometryVector6 << 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2;
        priorNoise = noiseModel::Diagonal::Variances(Vector6);
        odometryNoise = noiseModel::Diagonal::Variances(Vector6);

        matA0 = cv::Mat (5, 3, CV_32F, cv::Scalar::all(0));
        matB0 = cv::Mat (5, 1, CV_32F, cv::Scalar::all(-1));
        matX0 = cv::Mat (3, 1, CV_32F, cv::Scalar::all(0));

        // matA1为边缘特征的协方差矩阵
        matA1 = cv::Mat (3, 3, CV_32F, cv::Scalar::all(0));
        // matA1的特征值
        matD1 = cv::Mat (1, 3, CV_32F, cv::Scalar::all(0));
        // matA1的特征向量，对应于matD1存储
        matV1 = cv::Mat (3, 3, CV_32F, cv::Scalar::all(0));

        isDegenerate = false;
        matP = Eigen::Matrix<float, 6, 6>::Zero();

        laserCloudCornerFromMapDSNum = 0;
        laserCloudSurfFromMapDSNum = 0;
        laserCloudCornerLastDSNum = 0;
        laserCloudSurfLastDSNum = 0;
        // laserCloudOutlierLastDSNum = 0;
        laserCloudSurfTotalLastDSNum = 0;

        potentialLoopFlag = false;
        aLoopIsClosed = false;

        latestFrameID = 0;
    }

    void laserCloudCornerLastHandler(const sensor_msgs::PointCloud2ConstPtr& msg){
        timeLaserCloudCornerLast = msg->header.stamp.toSec();
        laserCloudCornerLast->clear();
        pcl::fromROSMsg(*msg, *laserCloudCornerLast);
        newLaserCloudCornerLast = true;
    }

    void laserCloudSurfLastHandler(const sensor_msgs::PointCloud2ConstPtr& msg){
        timeLaserCloudSurfLast = msg->header.stamp.toSec();
        laserCloudSurfLast->clear();
        pcl::fromROSMsg(*msg, *laserCloudSurfLast);
        newLaserCloudSurfLast = true;
    }

    void laserOdometryHandler(const nav_msgs::Odometry::ConstPtr& laserOdometry){
        timeLaserOdometry = laserOdometry->header.stamp.toSec();

        double roll, pitch, yaw;
        geometry_msgs::Quaternion geoQuat = laserOdometry->pose.pose.orientation;
        tf::Matrix3x3(tf::Quaternion(geoQuat.x, geoQuat.y, geoQuat.z, geoQuat.w)).getRPY(roll, pitch, yaw);
        transformSum[0] = roll;
        transformSum[1] = pitch;
        transformSum[2] = yaw;
        transformSum[3] = laserOdometry->pose.pose.position.x;
        transformSum[4] = laserOdometry->pose.pose.position.y;
        transformSum[5] = laserOdometry->pose.pose.position.z;

        q_transformSum.x() = laserOdometry->pose.pose.orientation.x;
        q_transformSum.y() = laserOdometry->pose.pose.orientation.y;
        q_transformSum.z() = laserOdometry->pose.pose.orientation.z;
        q_transformSum.w() = laserOdometry->pose.pose.orientation.w;
        v_transformSum.x() = laserOdometry->pose.pose.position.x;
        v_transformSum.y() = laserOdometry->pose.pose.position.y;
        v_transformSum.z() = laserOdometry->pose.pose.position.z;
        newLaserOdometry = true;

        Eigen::Vector3f v_transformIncre_, v_transformTobeMapped_;
        Eigen::Quaternionf q_transformIncre_, q_transformTobeMapped_;
        v_transformIncre_ = v_transformSum - v_transformBefMapped;
        q_transformIncre_ = q_transformSum * q_transformBefMapped.inverse();
        v_transformIncre_ = q_transformBefMapped.inverse() * v_transformIncre_;
        v_transformTobeMapped_ = v_transformAftMapped + q_transformAftMapped * v_transformIncre_;
        q_transformTobeMapped_ = q_transformAftMapped * q_transformIncre_;
        aftMappedOdomTrans.stamp_ = ros::Time().fromSec(timeLaserOdometry);
        aftMappedOdomTrans.setRotation(tf::Quaternion(q_transformTobeMapped_.x(),
                                                 q_transformTobeMapped_.y(), 
                                                 q_transformTobeMapped_.z(), 
                                                 q_transformTobeMapped_.w()));
        aftMappedOdomTrans.setOrigin(tf::Vector3(v_transformTobeMapped_.x(), 
                                                v_transformTobeMapped_.y(), 
                                                v_transformTobeMapped_.z()));
        tfBroadcaster.sendTransform(aftMappedOdomTrans);
    }

    // 将坐标转移到世界坐标系下,得到可用于建图的Lidar坐标，即修改了transformTobeMapped的值
    void transformAssociateToMap()
    {
        v_transformIncre = v_transformSum - v_transformBefMapped;
        q_transformIncre = q_transformSum * q_transformBefMapped.inverse();
        v_transformIncre = q_transformBefMapped.inverse() * v_transformIncre;

        // 遵守先平移后旋转的规矩
        q_transformTobeMapped = q_transformIncre * q_transformAftMapped;
        v_transformTobeMapped = v_transformAftMapped + q_transformAftMapped * v_transformIncre;

        // transformTobeMapped[3] = v_transformTobeMapped[0];
        // transformTobeMapped[4] = v_transformTobeMapped[1];
        // transformTobeMapped[5] = v_transformTobeMapped[2];
        // Eigen::Vector3f rotation = q_transformTobeMapped.toRotationMatrix().eulerAngles(0, 1, 2);
        // transformTobeMapped[0] = rotation[0];
        // transformTobeMapped[1] = rotation[1];
        // transformTobeMapped[2] = rotation[2];

        // 后面记得更新transformBefMapped
    }

    void updateTransformTobeMapped()
    {
        geometry_msgs::Quaternion geoQuat = tf::createQuaternionMsgFromRollPitchYaw
        (transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
        q_transformTobeMapped.x() = geoQuat.x;
        q_transformTobeMapped.y() = geoQuat.y;
        q_transformTobeMapped.z() = geoQuat.z;
        q_transformTobeMapped.w() = geoQuat.w;

        v_transformTobeMapped.x() = transformTobeMapped[3];
        v_transformTobeMapped.y() = transformTobeMapped[4];
        v_transformTobeMapped.z() = transformTobeMapped[5];
    }

    pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn,
                                                        const myPointTypePose* const tIn)
    {
        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
        PointType pointTo;

        // Eigen::AngleAxisf rollAngle(
        //     Eigen::AngleAxisf(tIn->roll, Eigen::Vector3f::UnitX())
        // );
        // Eigen::AngleAxisf pitchAngle(
        //     Eigen::AngleAxisf(tIn->pitch, Eigen::Vector3f::UnitY())
        // );
        // Eigen::AngleAxisf yawAngle(
        //     Eigen::AngleAxisf(tIn->yaw, Eigen::Vector3f::UnitZ())
        // ); 
        Eigen::Quaternionf q_tIn(tIn->qw, tIn->qx, tIn->qy, tIn->qz);
        Eigen::Vector3f v_tIn(tIn->x, tIn->y, tIn->z);

        for(int i=0; i<cloudIn->size(); i++)
        {
            pointTo = cloudIn->points[i];
            Eigen::Vector3f point(cloudIn->points[i].x,
                                    cloudIn->points[i].y,
                                    cloudIn->points[i].z);
            point = q_tIn * point + v_tIn;

            pointTo.x = point.x();
            pointTo.y = point.y();
            pointTo.z = point.z();

            cloudOut->push_back(pointTo);
        }
        return cloudOut;
    }

    void extractSurroundingKeyFrames(){
        if(cloudKeyPoses3D->points.empty() == true)
            return;
        
        if(loopClosureEnableFlag == true)
        {

        }
        else // 无回环
        {
            surroundingKeyPoses->clear();
            surroundingKeyPosesDS->clear();
            // 把所有关键帧位姿构造kd-tree
            kdtreeSurroundingKeyPoses->setInputCloud(cloudKeyPoses3D);
            // 进行半径surroundingKeyframeSearchRadius内的邻域搜索，
            // currentRobotPosPoint：需要查询的点，
            // pointSearchInd：搜索完的邻域点对应的索引
            // pointSearchSqDis：搜索完的每个领域点点与传讯点之间的欧式距离
            // 0：返回的邻域个数，为0表示返回全部的邻域点
            kdtreeSurroundingKeyPoses->radiusSearch(currentRobotPosPoint, 
                                                    (double)surroundingKeyframeSearchRadius, 
                                                    pointSearchInd,
                                                    pointSearchSqDis,
                                                    0);
            // std::cout<<"cloudKeyPoses3D : "<<cloudKeyPoses3D->size()<<std::endl;
            // 遍历找到的临近点,取对应的位姿xyz坐标到surroundingKeyPoses
            for(int i=0; i<pointSearchInd.size(); i++)
                surroundingKeyPoses->points.push_back(cloudKeyPoses3D->points[pointSearchInd[i]]);
            // 对临近位姿进行滤波(1m,1m,1m)
            downSizeFilterSurroundingKeyPoses.setInputCloud(surroundingKeyPoses);
            downSizeFilterSurroundingKeyPoses.filter(*surroundingKeyPosesDS);

            int numSurroundingPosesDS = surroundingKeyPosesDS->points.size();
            // 遍历已经存在的keypose
            // 如果最新的最近临列表surroundingKeyPosesDS[]里面没有这些已存在的keypose，则把这些已存在的keypose剔除
            for(int i=0; i<surroundingExistingKeyPosesID.size(); i++) // surroundingExistingKeyPosesID相当于管理局部地图的滑窗
            {
                bool existingFlag = false;
                // 同时遍历上面刚刚找到的滤波后的临近位姿
                for(int j=0; j<numSurroundingPosesDS; j++)
                {
                    // cloudKeyPoses3D->point[].intensity表示点的索引
                    // 双重循环，不断对比surroundingExistingKeyPosesID[i]和surroundingKeyPosesDS的点的index
                    // 如果能够找到一样的，说明存在相同的关键位姿，表示找到，跳出循环
                    if(surroundingExistingKeyPosesID[i] == (int)surroundingKeyPosesDS->points[j].intensity)
                    {
                        existingFlag = true;
                        break;
                    }
                }
                // 如果surroundingKeyPosesDS[]里面没有surroundingExistingKeyPosesID[i]这个点的索引
                if(existingFlag == false)
                {
                    // 没有找到相同的关键点，那么把这个点从各个队列中删除:
                    // 1. surroundingExistingKeyPosesID[i]
                    // 2. surroundingCornerCloudKeyFrames[i]
                    // 3. surroundingSurfCloudKeyFrames[i]
                    // 4. surroundingOutlierCloudKeyFrames[i]
                    // 否则的话，existingFlag为true，该关键点就将它留在队列中
                    surroundingExistingKeyPosesID.   erase(surroundingExistingKeyPosesID.   begin() + i);
                    surroundingCornerCloudKeyFrames. erase(surroundingCornerCloudKeyFrames. begin() + i);
                    surroundingSurfCloudKeyFrames.   erase(surroundingSurfCloudKeyFrames.   begin() + i);
                    // surroundingOutlierCloudKeyFrames.erase(surroundingOutlierCloudKeyFrames.begin() + i);
                    --i;
                }
            }
            // std::cout<<"surroundingExistingKeyPosesID : "<<surroundingExistingKeyPosesID.size()<<std::endl;

            // 上一个两重for循环主要用于删除数据，此处的两重for循环用来添加数据
            // 遍历最新的最近临列表，如果已存在keypose容器中没有新找到的临近点，则添加进去容器
            for(int i=0; i<numSurroundingPosesDS; i++)
            {
                bool existingFlag = false;
                // 遍历已存在的keypose容器
                for (auto iter = surroundingExistingKeyPosesID.begin(); iter != surroundingExistingKeyPosesID.end(); ++iter)
                {
                    // *iter就是不同的cloudKeyPoses3D->points.size(),
                    // 把surroundingExistingKeyPosesID内没有对应的点放进一个队列里
                    // 这个队列专门存放周围存在的关键帧，但是和surroundingExistingKeyPosesID的点没有对应的，也就是新的点
                    if((*iter) == (int)surroundingKeyPosesDS->points[i].intensity)
                    {
                        existingFlag = true;
                        break;
                    }
                }
                // 已存在容器中，则不添加，否则，添加到各个容器：
                // 1. surroundingExistingKeyPosesID[]
                // 2. surroundingCornerCloudKeyFrames[]
                // 3. surroundingSurfCloudKeyFrames[]
                // 4. surroundingOutlierCloudKeyFrames[]
                if(existingFlag == true)
                    continue;
                else
                {
                    int thisKeyInd = (int)surroundingKeyPosesDS->points[i].intensity;
                    // 取对应关键帧的优化后的位姿（准备把对应的点云转换到世界坐标系上）
                    myPointTypePose thisTransformation = cloudKeyPoses6D->points[thisKeyInd];

                    // updateTransformPointCloudSinCos(&thisTransformation);
                    surroundingExistingKeyPosesID.   push_back(thisKeyInd);
                    // 先把边缘点、平面点变换到世界坐标系(相机导航坐标系n')
                    // 然后push
                    surroundingCornerCloudKeyFrames. push_back(transformPointCloud(cornerCloudKeyFrames[thisKeyInd], &thisTransformation));
                    surroundingSurfCloudKeyFrames.   push_back(transformPointCloud(surfCloudKeyFrames[thisKeyInd], &thisTransformation));
                    // surroundingOutlierCloudKeyFrames.push_back(transformPointCloud(outlierCloudKeyFrames[thisKeyInd], &thisTransformation));
                }
            }
            // std::cout<<"surroundingCornerCloudKeyFrames : "<<surroundingCornerCloudKeyFrames.size()<<std::endl;
            // 重新遍历已存在的keypose容器，对最近临keypose进行地图拼接
            // 得到局部地图 [世界坐标系(相机导航坐标系n')]
            for(int i=0; i<surroundingExistingKeyPosesID.size(); i++)
            {
                *laserCloudCornerFromMap += *surroundingCornerCloudKeyFrames[i];
                *laserCloudSurfFromMap   += *surroundingSurfCloudKeyFrames[i];
            }
        }
        // 进行两次下采样
        // 最后的输出结果是laserCloudCornerFromMapDS和laserCloudSurfFromMapDS
        downSizeFilterCorner.setInputCloud(laserCloudCornerFromMap);
        downSizeFilterCorner.filter(*laserCloudCornerFromMapDS);
        laserCloudCornerFromMapDSNum = laserCloudCornerFromMapDS->points.size();

        downSizeFilterSurf.setInputCloud(laserCloudSurfFromMap);
        downSizeFilterSurf.filter(*laserCloudSurfFromMapDS);
        laserCloudSurfFromMapDSNum = laserCloudSurfFromMapDS->points.size();
    }

    void downsampleCurrentScan(){
        // 最新一帧的less sharp特征
        laserCloudCornerLastDS->clear();
        downSizeFilterCorner.setInputCloud(laserCloudCornerLast);
        downSizeFilterCorner.filter(*laserCloudCornerLastDS);
        laserCloudCornerLastDSNum = laserCloudCornerLastDS->points.size();

        // 最新一帧的less surf特征
        laserCloudSurfLastDS->clear();
        downSizeFilterSurf.setInputCloud(laserCloudSurfLast);
        downSizeFilterSurf.filter(*laserCloudSurfLastDS);
        laserCloudSurfLastDSNum = laserCloudSurfLastDS->points.size();

        laserCloudSurfTotalLast->clear();
        laserCloudSurfTotalLastDS->clear();
        // *laserCloudSurfTotalLast = *laserCloudSurfLastDS;
        *laserCloudSurfTotalLastDS = *laserCloudSurfLastDS;
        // downSizeFilterSurf.setInputCloud(laserCloudSurfTotalLast);
        // downSizeFilterSurf.filter(*laserCloudSurfTotalLastDS);
        laserCloudSurfTotalLastDSNum = laserCloudSurfTotalLastDS->points.size();

        // std::cout<<"laserCloudCornerLastDSNum : "<<laserCloudCornerLastDSNum<<", ";
        // std::cout<<"laserCloudSurfTotalLastDSNum : "<<laserCloudSurfTotalLastDSNum<<std::endl;
    }

    void transformUpdate()
    {
        // 把当前帧里程计值记录到transformBefMapped[]
        // 把优化后的位姿记录到transformAftMapped[]
        q_transformBefMapped = q_transformSum;
        v_transformBefMapped = v_transformSum;

        q_transformAftMapped = q_transformTobeMapped;
        v_transformAftMapped = v_transformTobeMapped;
    }

    void pointAssociateToMap(PointType const * const pi, PointType * const po)
    {
        // 进行6自由度的变换，先进行旋转，然后再平移
        // 主要进行坐标变换，将局部坐标转换到全局坐标中去
        Eigen::Vector3f point(pi->x, pi->y, pi->z);
        point = q_transformTobeMapped * point + v_transformTobeMapped;
        po->x = point.x();
        po->y = point.y();
        po->z = point.z();
    }

    void cornerOptimization(int iterCount)
    {
        // 遍历当前帧的边缘点集
        for(int i=0; i<laserCloudCornerLastDSNum; i++)
        {
            // 取点
            // laserCloudCornerLastDS: 当前帧扫描结束时相机坐标系的点
            pointOri = laserCloudCornerLastDS->points[i];
            // 进行坐标变换,转换到全局坐标中去（世界坐标系）
            // 输入是pointOri，输出是pointSel
            // pointSel：边缘点投影到世界坐标系的投影点
            pointAssociateToMap(&pointOri, &pointSel);

            // 进行5邻域搜索，
            // pointSel为需要搜索的点，
            // pointSearchInd搜索完的邻域对应的索引
            // pointSearchSqDis 邻域点与查询点之间的距离
            kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

            // 只有当最远的那个邻域点的距离pointSearchSqDis[4]小于1m时才进行下面的计算
            // 以下部分的计算是在计算点集的协方差矩阵，Zhang Ji的论文中有提到这部分
            if(pointSearchSqDis[4] < 1.0)
            {
                // 求五个点的均值
                std::vector<Eigen::Vector3f> nearCorners;
                Eigen::Vector3f center(0.0, 0.0, 0.0);
                for(int j=0; j<5; j++)
                {
                    Eigen::Vector3f tmp(
                        laserCloudCornerFromMapDS->points[pointSearchInd[j]].x,
                        laserCloudCornerFromMapDS->points[pointSearchInd[j]].y,
                        laserCloudCornerFromMapDS->points[pointSearchInd[j]].z
                    );
                    center = center + tmp;
                    nearCorners.push_back(tmp);
                }
                center = center / 5.0;

                // 求五个近邻点的分布协方差
                Eigen::Matrix3f covMat = Eigen::Matrix3f::Zero();
                for(int j=0; j<5; j++)
                {
                    Eigen::Vector3f tmpZeroMean = nearCorners[j] - center;
                    covMat = covMat + tmpZeroMean * tmpZeroMean.transpose();
                }

                // SVD分解协方差，如果最大特征值远大于其余两个特征值，说明分布成直线
                // 最大特征值对应特征向量即为直线的方向
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> saes(covMat);
                // note Eigen library sort eigenvalues in increasing order
                Eigen::Vector3f unit_direction = saes.eigenvectors().col(2);

                if(saes.eigenvalues()[2] > 3 * saes.eigenvalues()[1])
                {
                    // 根据直线方向和中心点，在直线上找两个点
                    Eigen::Vector3f point_on_line = center;
                    // 提取到的直线上的两个点，用于确定方向
                    Eigen::Vector3f tmp_a, tmp_b;
                    tmp_a = 0.1 * unit_direction + point_on_line;
                    tmp_b = -0.1 * unit_direction + point_on_line; 
                    //选择的特征点记为O，kd-tree最近距离点记为A，另一个最近距离点记为B

                    Eigen::Vector3f tmp_p(pointSel.x, pointSel.y, pointSel.z);
                    // 计算点线距离
                    Eigen::Matrix<float, 3, 1> nu = (tmp_p - tmp_a).cross(tmp_p - tmp_b); //(叉乘)
                    Eigen::Matrix<float, 3, 1> de = tmp_a - tmp_b;
                    Eigen::Matrix<float, 3, 1> ld = de.cross(nu);

                    ld.normalize();

                    float la = ld.x();
                    float lb = ld.y();
                    float lc = ld.z();
                    float ld2 = nu.norm() / de.norm();

                    //权重计算，距离越大权重越小，距离越小权重越大，得到的权重范围<=1
                    float s;
                    //增加权重，距离越远，影响影子越小
                    s = 1 - 0.9 * fabs(ld2);

                    if (s > 0.2) {
                        coeff.x = s * la;
                        coeff.y = s * lb;
                        coeff.z = s * lc;
                        coeff.intensity = s * ld2;
                        
                        // 当前帧的特征点
                        laserCloudOri->push_back(pointOri);
                        // 特征点指到特征直线的垂线方向，其中intensity存储了点到直线的距离，也就是残差
                        coeffSel->push_back(coeff);
                    }
                }
            }
        }
    }

    void surfOptimization(int iterCount)
    {
        for(int i=0; i<laserCloudSurfLastDSNum; i++)
        {
            // 取当前帧的平面点
            pointOri = laserCloudSurfLastDS->points[i];
            // 转换到世界坐标
            pointAssociateToMap(&pointOri, &pointSel);
            // 在局部地图中，检索5个最近邻点
            kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

            Eigen::Matrix<float, 5, 3> matA0;
            Eigen::Matrix<float, 5, 1> matB0 = -1 * Eigen::Matrix<float, 5, 1>::Ones();
            if(pointSearchSqDis[4] < 1.0)
            {
                // 找到5个近邻点组成的平面
                // 1. SVD分解求解
                // 通过协方差矩阵的特征向量，找到平面的法向量
                // 首先要通过特征值判断是否形成平面：一个特征值远小于另外两个特征值，可以参考NDT算法解析
                // 最小特征值对应的特征向量即为平面的法向量
                // 2. QR分解求解
                // 当然，也可以通过求解最小二乘问题AX=b得到
                // 但是QR分解一般比SVD分解快，所以我们选择使用QR分解
                for(int j=0; j<5; j++)
                {
                    matA0(j, 0) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].x;
                    matA0(j, 1) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].y;
                    matA0(j, 2) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].z;
                }
                // 计算平面的法向量
                Eigen::Vector3f norm = matA0.colPivHouseholderQr().solve(matB0);
                double negative_OA_dot_norm = 1/norm.norm();
                norm.normalize(); // 归一化

                // 将5个近邻点代入方程，验证
                bool planeValid = true;
                for(int j=0; j<5; j++)
                {
                    // if OX * n > 0.2, then plane is not fit well
                    double ox_n = fabs(norm(0)*laserCloudSurfFromMapDS->points[pointSearchInd[j]].x +
                                        norm(1)*laserCloudSurfFromMapDS->points[pointSearchInd[j]].y +
                                        norm(2)*laserCloudSurfFromMapDS->points[pointSearchInd[j]].z + 
                                        negative_OA_dot_norm);
                    if(ox_n > 0.2)
                    {
                        planeValid = false;
                        break;
                    }
                }

                if(planeValid)
                {
                    Eigen::Vector3f tmp_a(laserCloudSurfFromMapDS->points[pointSearchInd[0]].x, 
                                            laserCloudSurfFromMapDS->points[pointSearchInd[0]].y, 
                                            laserCloudSurfFromMapDS->points[pointSearchInd[0]].z);
                    Eigen::Vector3f tmp_b(laserCloudSurfFromMapDS->points[pointSearchInd[2]].x, 
                                            laserCloudSurfFromMapDS->points[pointSearchInd[2]].y, 
                                            laserCloudSurfFromMapDS->points[pointSearchInd[2]].z);
                    Eigen::Vector3f tmp_c(laserCloudSurfFromMapDS->points[pointSearchInd[4]].x, 
                                            laserCloudSurfFromMapDS->points[pointSearchInd[4]].y, 
                                            laserCloudSurfFromMapDS->points[pointSearchInd[4]].z);

                    Eigen::Vector3f tmp_p(pointSel.x, pointSel.y, pointSel.z);

                    Eigen::Vector3f ld = (tmp_b - tmp_a).cross(tmp_c - tmp_a); // 平面法向量
                    ld.normalize();
                    // 确保方向向量是由面指向点
                    // if(ld.dot(tmp_a - tmp_p)>0)
                    //     ld = -ld;

                    // 距离不要取绝对值
                    double pd2 = ld.dot(tmp_p - tmp_a);

                    // std::cout<<"pd2 = "<<pd2<<", ";

                    double pa = ld.x();
                    double pb = ld.y();
                    double pc = ld.z();

                    // if(pd2 > (10 * maxDis) && i>0)
                    //     break;
                    // maxDis = std::max(maxDis, pd2);
                    float s;
                    // 加上影响因子
                    s = 1 - 0.9 * fabs(pd2)/ sqrt(sqrt(pointOri.x * pointOri.x
                            + pointOri.y * pointOri.y + pointOri.z * pointOri.z));

                    if (s > 0.2) {
                        // [x,y,z]是整个平面的单位法量
                        // intensity是平面外一点到该平面的距离
                        coeff.x = s * pa;
                        coeff.y = s * pb;
                        coeff.z = s * pc;
                        coeff.intensity = s * pd2;

                        // 未经变换的点放入laserCloudOri队列，距离，法向量值放入coeffSel
                        laserCloudOri->push_back(pointOri);
                        coeffSel->push_back(coeff);
                    }
                }
            }
        }
    }

    bool LMOptimization(int iterCount)
    {
        int pointSelNum = laserCloudOri->points.size();

        // std::cout<<"pointSelNum : "<<pointSelNum<<std::endl;

        if(pointSelNum < 50)
            return false;

        Eigen::Matrix<float, 3, 1> v_pointOri_bk1;

        Eigen::Matrix<float, 1, 6> j_n = Eigen::Matrix<float, 1, 6>::Zero();
        Eigen::Matrix<float, 6, 6> H = Eigen::Matrix<float, 6, 6>::Zero();
        Eigen::Matrix<float, 6, 1> b = Eigen::Matrix<float, 6, 1>::Zero();

        Eigen::Matrix<float, 6, 1> Delta_x = Eigen::Matrix<float, 6, 1>::Zero();
        Eigen::Quaternionf q_Delta_x = Eigen::Quaternionf::Identity();
        Eigen::Vector3f v_Delta_x = Eigen::Vector3f::Zero();

        // #define EULER_DERIVATION
        #ifdef EULER_DERIVATION
        // Eigen::Vector3f e_transformTobeMapped = q_transformTobeMapped.toRotationMatrix().eulerAngles(0, 1, 2);
        double roll, pitch, yaw;
        tf::Matrix3x3(tf::Quaternion(q_transformTobeMapped.x(), 
                                      q_transformTobeMapped.y(), 
                                      q_transformTobeMapped.z(), 
                                      q_transformTobeMapped.w())).getRPY(roll, pitch, yaw);
        
        transformTobeMapped[0] = roll;
        transformTobeMapped[1] = pitch;
        transformTobeMapped[2] = yaw;
        transformTobeMapped[3] = v_transformTobeMapped.x();
        transformTobeMapped[4] = v_transformTobeMapped.y();
        transformTobeMapped[5] = v_transformTobeMapped.z();
        
        float s1 = sin(roll);
        float c1 = cos(roll);
        float s2 = sin(pitch);
        float c2 = cos(pitch);
        float s3 = sin(yaw);
        float c3 = cos(yaw);
        Eigen::Matrix<float, 3, 3> dRdrx;
        Eigen::Matrix<float, 3, 3> dRdry;
        Eigen::Matrix<float, 3, 3> dRdrz;
        dRdrx(0, 0) = 0.0;
        dRdrx(0, 1) = 0.0;
        dRdrx(0, 2) = 0.0;
        dRdrx(1, 0) = -s3*s1+s2*c3*c1;
        dRdrx(1, 1) = -c3*s1-s2*s3*c1;
        dRdrx(1, 2) = -c2*c1;
        dRdrx(2, 0) = s3*c1+s2*c3*s1;
        dRdrx(2, 1) = c3*c1-s2*s3*s1;
        dRdrx(2, 2) = -c2*s1;

        dRdry(0, 0) = -c3*s2;
        dRdry(0, 1) = -s3*s2;
        dRdry(0, 2) = c2;
        dRdry(1, 0) = c3*s1*c2;
        dRdry(1, 1) = -s1*s3*c2;
        dRdry(1, 2) = s1*s2;
        dRdry(2, 0) = -c1*c3*c2;
        dRdry(2, 1) = c1*s3*c2;
        dRdry(2, 2) = -c1*s2;

        dRdrz(0, 0) = -c2*s3;
        dRdrz(0, 1) = -c2*c3;
        dRdrz(0, 2) = 0.0;
        dRdrz(1, 0) = c1*c3-s1*s2*s3;
        dRdrz(1, 1) = -c1*s3-s1*s2*c3;
        dRdrz(1, 2) = 0.0;
        dRdrz(2, 0) = s1*c3+c1*s2*s3;
        dRdrz(2, 1) = -s1*s3+c1*s2*c3;
        dRdrz(2, 2) = 0.0;
        #endif

        for(int i=0; i<pointSelNum; i++)
        {
            // 当前点，在b_k+1坐标系
            pointOri = laserCloudOri->points[i];
            // 由当前点指向直线特征的垂线方向向量，其中intensity为距离值
            coeff = coeffSel->points[i];

            // 1. 计算G函数，通过估计的变换transformCur将pointOri转换到b_k坐标系
            v_pointOri_bk1(0, 0) = pointOri.x;
            v_pointOri_bk1(1, 0) = pointOri.y;
            v_pointOri_bk1(2, 0) = pointOri.z;
            // 2. dD/dG = (la, lb, lc)
            Eigen::Matrix<float, 1, 3> dDdG;
            dDdG << coeff.x, coeff.y, coeff.z;
            Eigen::Matrix<float, 1, 3> dDdR;
            #ifdef EULER_DERIVATION
            dDdR(0, 0) = dDdG * dRdrx * v_pointOri_bk1;
            dDdR(0, 1) = dDdG * dRdry * v_pointOri_bk1;
            dDdR(0, 2) = dDdG * dRdrz * v_pointOri_bk1;
            #else
            // 3. 将transformCur转成R，然后计算(-Rp)^
            Eigen::Matrix3f neg_Rp_sym;
            // 左扰动
            // anti_symmetric(q_transformTobeMapped.toRotationMatrix()*v_pointOri_bk1, neg_Rp_sym);
            // neg_Rp_sym = -neg_Rp_sym;
            // 右扰动
            anti_symmetric(v_pointOri_bk1, neg_Rp_sym);
            neg_Rp_sym = -q_transformTobeMapped.toRotationMatrix()*neg_Rp_sym;

            // dDdG = dDdG * q_transformTobeMapped.toRotationMatrix();
            // anti_symmetric(v_pointOri_bk1, neg_Rp_sym);
            // neg_Rp_sym = -q_transformTobeMapped.toRotationMatrix()*neg_Rp_sym;

            // 4. 计算(dD/dG)*(-Rp)^得到关于旋转的雅克比，取其中的yaw部分，记为j_yaw
            dDdR = dDdG * neg_Rp_sym;
            #endif
            // 5. 计算关于平移的雅克比，即为(dD/dG)，取其中的x,y部分，记为j_x,j_y
            // 6. 组织该点的雅克比：[j_yaw,j_x,j_y]
            j_n(0, 0) = dDdR(0, 0);
            j_n(0, 1) = dDdR(0, 1);
            j_n(0, 2) = dDdR(0, 2);
            j_n(0, 3) = dDdG(0, 0);
            j_n(0, 4) = dDdG(0, 1);
            j_n(0, 5) = dDdG(0, 2);

            // 7. 该点的残差值,f(x)=coeff.intensity
            float f_n = coeff.intensity;
            // 8. 组织近似海森矩阵H += J^T*J
            H = H + j_n.transpose() * j_n;
            // 9. 组织方程右边：b += -J^T*f(x)
            b = b - f_n * j_n.transpose();
        }
        Delta_x = H.colPivHouseholderQr().solve(b);

        if(iterCount == 0)
        {
            Eigen::Matrix<float, 1, 6> matE = Eigen::Matrix<float, 1, 6>::Zero();
            Eigen::Matrix<float, 6, 6> matV = Eigen::Matrix<float, 6, 6>::Zero();
            Eigen::Matrix<float, 6, 6> matV2 = Eigen::Matrix<float, 6, 6>::Zero();

            // 计算At*A的特征值和特征向量
            // 特征值存放在matE，特征向量matV
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float, 6, 6>> eigenSolver(H);
            matE = eigenSolver.eigenvalues().real();
            matV = eigenSolver.eigenvectors().real();

            matV2 = matV;
            // 退化的具体表现是指什么？
            isDegenerate = false;
            float eignThre[6] = {100, 100, 100, 100, 100, 100};
            for (int i = 5; i >= 0; i--) {
                if (matE(0, i) < eignThre[i]) {
                    for (int j = 0; j < 6; j++) {
                        matV2(i, j) = 0;
                    }
                    // 存在比100小的特征值则出现退化
                    isDegenerate = true;
                } else {
                    break;
                }
            }
            matP = matV.inverse() * matV2;
            // std::cout<<iterCount<<"] isDegenerate : "<<isDegenerate<<std::endl;
        }

        if (isDegenerate) {
            Eigen::Matrix<float, 6, 1> matX2 = Eigen::Matrix<float, 6, 1>::Zero();
            // matX.copyTo(matX2);
            matX2 = Delta_x;
            Delta_x = matP * matX2;
        }

        // 更新
        v_Delta_x[0] = Delta_x(3, 0);
        v_Delta_x[1] = Delta_x(4, 0);
        v_Delta_x[2] = Delta_x(5, 0);
        #ifdef EULER_DERIVATION
        Eigen::AngleAxisf rollAngle = Eigen::AngleAxisf(Delta_x(0, 0), 
                                                        Eigen::Vector3f::UnitX());
        Eigen::AngleAxisf pitchAngle = Eigen::AngleAxisf(Delta_x(1, 0), 
                                                        Eigen::Vector3f::UnitY());
        Eigen::AngleAxisf yawAngle = Eigen::AngleAxisf(Delta_x(2, 0), 
                                                        Eigen::Vector3f::UnitZ());
        q_Delta_x = rollAngle*pitchAngle*yawAngle;
        #else
        Eigen::Vector3f axis = Delta_x.block<3, 1>(0, 0);
        axis.normalize();
        Eigen::AngleAxisf Angle = Eigen::AngleAxisf(Delta_x.block<3, 1>(0, 0).norm(), axis);
        q_Delta_x = Angle;

        v_transformTobeMapped = v_transformTobeMapped + v_Delta_x;
        // 右扰动
        q_transformTobeMapped = q_transformTobeMapped * q_Delta_x;
        q_transformTobeMapped.normalize();
        // 左扰动
        // q_transformTobeMapped = q_Delta_x * q_transformTobeMapped;
        #endif

        // std::cout<<iterCount<<"] Delta_x = "<<Delta_x.head<3>().transpose()/PI*180<<std::endl;

        #ifdef EULER_DERIVATION
        transformTobeMapped[0] += Delta_x(0, 0);
        transformTobeMapped[1] += Delta_x(1, 0);
        transformTobeMapped[2] += Delta_x(2, 0);
        transformTobeMapped[3] += Delta_x(3, 0);
        transformTobeMapped[4] += Delta_x(4, 0);
        transformTobeMapped[5] += Delta_x(5, 0);
        updateTransformTobeMapped();
        #endif
        
        // Eigen::Vector3f tmp = q_transformTobeMapped.toRotationMatrix().eulerAngles(2, 1, 0);

        // std::cout<<"transformTobeMapped[2] : "<<tmp[0]/PI*180<<std::endl;

        float deltaR = sqrt(
                            pow(pcl::rad2deg(Delta_x(0, 0)), 2) +
                            pow(pcl::rad2deg(Delta_x(1, 0)), 2) +
                            pow(pcl::rad2deg(Delta_x(2, 0)), 2));
        float deltaT = sqrt(
                            pow(Delta_x(3, 0) * 100, 2) +
                            pow(Delta_x(4, 0) * 100, 2) +
                            pow(Delta_x(5, 0) * 100, 2));

        // 旋转或者平移量足够小就停止这次迭代过程
        if (deltaR < 0.05 && deltaT < 0.05) {
            return true;
        }
        return false;
    }

    void scan2MapOptimization(){
        // laserCloudCornerFromMapDSNum是extractSurroundingKeyFrames()函数最后降采样得到的局部地图coner点云数
        // laserCloudSurfFromMapDSNum是extractSurroundingKeyFrames()函数降采样得到的局部地图surface点云数
        // 如果局部地图的点数>阈值
        if (laserCloudCornerFromMapDSNum > 10 && laserCloudSurfFromMapDSNum > 100) 
        {
            // 局部地图构造kt-tree
            // laserCloudCornerFromMapDS和laserCloudSurfFromMapDS的来源有2个：
            // 当有闭环时，来源是：recentCornerCloudKeyFrames，没有闭环时，来源surroundingCornerCloudKeyFrames
            kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMapDS);
            kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMapDS);

            // 开始迭代优化
            for(int iterCount = 0; iterCount < 10; iterCount++)
            {
                // 用for循环控制迭代次数，最多迭代10次
                laserCloudOri->clear();
                coeffSel->clear();

                cornerOptimization(iterCount);
                surfOptimization(iterCount);

                if (LMOptimization(iterCount) == true)
                    break;    
            }
            // 迭代结束更新相关的转移矩阵
            transformUpdate();
        }
    }

    void saveKeyFramesAndFactor()
    {
        // 取优化后的位姿transformAftMapped[]
        // 储存到currentRobotPosPoint
        currentRobotPosPoint.x = v_transformAftMapped[0];
        currentRobotPosPoint.y = v_transformAftMapped[1];
        currentRobotPosPoint.z = v_transformAftMapped[2];

        // Eigen::Vector3f e_transformAftMapped;
        // e_transformAftMapped = 
        //         (previousRobotAttPoint.inverse()*q_transformAftMapped).toRotationMatrix().eulerAngles(0, 1, 2);
        // if(fabs(e_transformAftMapped[0]) > 1.57)
        //     e_transformAftMapped[0] = fabs(fabs(e_transformAftMapped[0])-PI);
        // if(fabs(e_transformAftMapped[1]) > 1.57)
        //     e_transformAftMapped[1] = fabs(fabs(e_transformAftMapped[1])-PI);
        // if(fabs(e_transformAftMapped[2]) > 1.57)
        //     e_transformAftMapped[2] = fabs(fabs(e_transformAftMapped[2])-PI);
        // std::cout<<"e_transformAftMapped : "<<e_transformAftMapped.sum()<<std::endl;

        // 如果两次优化之间欧式距离<0.3，则不保存，不作为关键帧
        bool saveThisKeyFrame = true;
        if (sqrt((previousRobotPosPoint.x-currentRobotPosPoint.x)*(previousRobotPosPoint.x-currentRobotPosPoint.x)
                +(previousRobotPosPoint.y-currentRobotPosPoint.y)*(previousRobotPosPoint.y-currentRobotPosPoint.y)
                +(previousRobotPosPoint.z-currentRobotPosPoint.z)*(previousRobotPosPoint.z-currentRobotPosPoint.z)) < 0.3){
            saveThisKeyFrame = false;
        }

        // if(e_transformAftMapped.sum() > 0.17)
        //     saveThisKeyFrame = true;

        // 非关键帧 并且 非空，直接返回
        if (saveThisKeyFrame == false && !cloudKeyPoses3D->points.empty())
        	return;

        previousRobotPosPoint = currentRobotPosPoint;
        previousRobotAttPoint = q_transformAftMapped;

        #define USE_GTSAM
        #ifdef USE_GTSAM
        // 如果还没有关键帧
        if (cloudKeyPoses3D->points.empty()){
            // static Rot3 	RzRyRx (double x, double y, double z),Rotations around Z, Y, then X axes
            // RzRyRx依次按照z(transformTobeMapped[2])，y(transformTobeMapped[0])，x(transformTobeMapped[1])坐标轴旋转
            // Point3 (double x, double y, double z)  Construct from x(transformTobeMapped[5]), y(transformTobeMapped[3]), and z(transformTobeMapped[4]) coordinates. 
            // Pose3 (const Rot3 &R, const Point3 &t) Construct from R,t. 从旋转和平移构造姿态
            // NonlinearFactorGraph增加一个PriorFactor因子
            // 构造的因子是在 载体坐标系b系及导航坐标系n系下的
            // 第一帧，作为先验因子
            gtSAMgraph.add(PriorFactor<Pose3>(0, Pose3(Rot3(q_transformTobeMapped.cast<double>()),
                                                Point3(v_transformTobeMapped.cast<double>())), priorNoise));
            // initialEstimate的数据类型是Values,其实就是一个map，这里在0对应的值下面保存了一个Pose3
            initialEstimate.insert(0, Pose3(Rot3(q_transformTobeMapped.cast<double>()),
                                                Point3(v_transformTobeMapped.cast<double>())));
            q_transformLast = q_transformTobeMapped;
            v_transformLast = v_transformTobeMapped;

        }
        else{
            // 非第一帧，添加边
            gtsam::Pose3 poseFrom = Pose3(Rot3(q_transformLast.cast<double>()),
                                            Point3(v_transformLast.cast<double>()));
            gtsam::Pose3 poseTo   = Pose3(Rot3(q_transformTobeMapped.cast<double>()),
                                            Point3(v_transformTobeMapped.cast<double>()));
			
            // 构造函数原型:BetweenFactor (Key key1, Key key2, const VALUE &measured, const SharedNoiseModel &model)
            gtSAMgraph.add(BetweenFactor<Pose3>(cloudKeyPoses3D->points.size()-1, 
                                                cloudKeyPoses3D->points.size(), 
                                                poseFrom.between(poseTo), odometryNoise));
            initialEstimate.insert(cloudKeyPoses3D->points.size(), poseTo);
        }
        // gtsam::ISAM2::update函数原型:
        // ISAM2Result gtsam::ISAM2::update	(	const NonlinearFactorGraph & 	newFactors = NonlinearFactorGraph(),
        // const Values & 	newTheta = Values(),
        // const std::vector< size_t > & 	removeFactorIndices = std::vector<size_t>(),
        // const boost::optional< FastMap< Key, int > > & 	constrainedKeys = boost::none,
        // const boost::optional< FastList< Key > > & 	noRelinKeys = boost::none,
        // const boost::optional< FastList< Key > > & 	extraReelimKeys = boost::none,
        // bool 	force_relinearize = false )	
        // gtSAMgraph是新加到系统中的因子
        // initialEstimate是加到系统中的新变量的初始点
        isam->update(gtSAMgraph, initialEstimate);
        // update 函数为什么需要调用两次？
        // isam2用法
        isam->update();

        // 删除内容?
        gtSAMgraph.resize(0);
		initialEstimate.clear();

        PointType thisPose3D;
        myPointTypePose thisPose6D;
        Pose3 latestEstimate;

        // Compute an estimate from the incomplete linear delta computed during the last update.
        isamCurrentEstimate = isam->calculateEstimate();
        latestEstimate = isamCurrentEstimate.at<Pose3>(isamCurrentEstimate.size()-1);

        Eigen::Vector3d v_latestEstimate = latestEstimate.translation();
        Eigen::Quaterniond q_latestEstimate = latestEstimate.rotation().toQuaternion();
        // GTSAM优化过后的旋转务必进行归一化，否则会计算异常
        q_latestEstimate.normalize();

        // 取优化结果
        // 又回到相机坐标系c系以及导航坐标系n'
        thisPose3D.x = v_latestEstimate.x();
        thisPose3D.y = v_latestEstimate.y();
        thisPose3D.z = v_latestEstimate.z();
        thisPose3D.intensity = cloudKeyPoses3D->points.size();
        cloudKeyPoses3D->push_back(thisPose3D);

        thisPose6D.x = thisPose3D.x;
        thisPose6D.y = thisPose3D.y;
        thisPose6D.z = thisPose3D.z;
        thisPose6D.intensity = thisPose3D.intensity;
        thisPose6D.qx = q_latestEstimate.x();
        thisPose6D.qy = q_latestEstimate.y();
        thisPose6D.qz = q_latestEstimate.z();
        thisPose6D.qw = q_latestEstimate.w();
        thisPose6D.time = timeLaserOdometry;
        cloudKeyPoses6D->push_back(thisPose6D);

        // 更新transformAftMapped[]
        if (cloudKeyPoses3D->points.size() > 1){
            q_transformAftMapped = q_latestEstimate.cast<float>();
            v_transformAftMapped = v_latestEstimate.cast<float>();

            q_transformLast = q_transformAftMapped;
            v_transformLast = v_transformAftMapped;

            q_transformTobeMapped = q_transformAftMapped;
            v_transformTobeMapped = v_transformAftMapped;
        }
        #else
        PointType thisPose3D;
        myPointTypePose thisPose6D;

        // 取优化结果
        // 又回到相机坐标系c系以及导航坐标系n'
        thisPose3D.x = v_transformAftMapped.x();
        thisPose3D.y = v_transformAftMapped.y();
        thisPose3D.z = v_transformAftMapped.z();
        thisPose3D.intensity = cloudKeyPoses3D->points.size();
        cloudKeyPoses3D->push_back(thisPose3D);

        thisPose6D.x = thisPose3D.x;
        thisPose6D.y = thisPose3D.y;
        thisPose6D.z = thisPose3D.z;
        thisPose6D.intensity = thisPose3D.intensity;
        // thisPose6D.roll  = latestEstimate.rotation().roll();
        // thisPose6D.pitch = latestEstimate.rotation().pitch();
        // thisPose6D.yaw   = latestEstimate.rotation().yaw();
        thisPose6D.qx = q_transformAftMapped.x();
        thisPose6D.qy = q_transformAftMapped.y();
        thisPose6D.qz = q_transformAftMapped.z();
        thisPose6D.qw = q_transformAftMapped.w();
        thisPose6D.time = timeLaserOdometry;
        cloudKeyPoses6D->push_back(thisPose6D);

        // 更新transformAftMapped[]
        // 更新transformAftMapped[]
        if (cloudKeyPoses3D->points.size() > 1){
            // q_transformAftMapped = latestEstimate.rotation().toQuaternion().cast<float>();
            // v_transformAftMapped = latestEstimate.translation().cast<float>();

            q_transformLast = q_transformAftMapped;
            v_transformLast = v_transformAftMapped;

            q_transformTobeMapped = q_transformAftMapped;
            v_transformTobeMapped = v_transformAftMapped;
        }
        #endif

        // 把新帧作为keypose，保存对应的边缘点、平面点
        pcl::PointCloud<PointType>::Ptr thisCornerKeyFrame(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr thisOutlierKeyFrame(new pcl::PointCloud<PointType>());

        // PCL::copyPointCloud(const pcl::PCLPointCloud2 &cloud_in,pcl::PCLPointCloud2 &cloud_out )   
        pcl::copyPointCloud(*laserCloudCornerLastDS,  *thisCornerKeyFrame);
        pcl::copyPointCloud(*laserCloudSurfLastDS,    *thisSurfKeyFrame);

        cornerCloudKeyFrames.push_back(thisCornerKeyFrame);
        surfCloudKeyFrames.push_back(thisSurfKeyFrame);
    }

    void publishTF(){
        // geometry_msgs::Quaternion geoQuat = tf::createQuaternionMsgFromRollPitchYaw
        //                           (transformAftMapped[2], -transformAftMapped[0], -transformAftMapped[1]);

        odomAftMapped.header.stamp = ros::Time().fromSec(timeLaserOdometry);
        odomAftMapped.pose.pose.orientation.x = q_transformAftMapped.x();
        odomAftMapped.pose.pose.orientation.y = q_transformAftMapped.y();
        odomAftMapped.pose.pose.orientation.z = q_transformAftMapped.z();
        odomAftMapped.pose.pose.orientation.w = q_transformAftMapped.w();
        odomAftMapped.pose.pose.position.x = v_transformAftMapped.x();
        odomAftMapped.pose.pose.position.y = v_transformAftMapped.y();
        odomAftMapped.pose.pose.position.z = v_transformAftMapped.z();
        pubOdomAftMapped.publish(odomAftMapped);

        // std::cout<<"v_transformAftMapped : "<<v_transformAftMapped.transpose()<<std::endl;

        aftMappedTrans.stamp_ = ros::Time().fromSec(timeLaserOdometry);
        aftMappedTrans.setRotation(tf::Quaternion(q_transformAftMapped.x(),
                                                 q_transformAftMapped.y(), 
                                                 q_transformAftMapped.z(), 
                                                 q_transformAftMapped.w()));
        aftMappedTrans.setOrigin(tf::Vector3(v_transformAftMapped.x(), 
                                                v_transformAftMapped.y(), 
                                                v_transformAftMapped.z()));
        tfBroadcaster.sendTransform(aftMappedTrans);
    }

    void publishKeyPosesAndFrames(){
        // 发布关键帧位置
        if (pubKeyPoses.getNumSubscribers() != 0){
            sensor_msgs::PointCloud2 cloudMsgTemp;
            pcl::toROSMsg(*cloudKeyPoses3D, cloudMsgTemp);
            cloudMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);
            cloudMsgTemp.header.frame_id = "/camera_init";
            pubKeyPoses.publish(cloudMsgTemp);
        }
        // 发布局部地图
        if (pubRecentKeyFrames.getNumSubscribers() != 0){
            sensor_msgs::PointCloud2 cloudMsgTemp;
            pcl::toROSMsg(*laserCloudSurfFromMapDS, cloudMsgTemp);
            cloudMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);
            cloudMsgTemp.header.frame_id = "/camera_init";
            pubRecentKeyFrames.publish(cloudMsgTemp);
        }
    }

    void clearCloud(){
        laserCloudCornerFromMap->clear();
        laserCloudSurfFromMap->clear();
        laserCloudCornerFromMapDS->clear();
        laserCloudSurfFromMapDS->clear();   
    }

    void run(){

        if (newLaserCloudCornerLast  && std::abs(timeLaserCloudCornerLast  - timeLaserOdometry) < 0.005 &&
            newLaserCloudSurfLast    && std::abs(timeLaserCloudSurfLast    - timeLaserOdometry) < 0.005 &&
            newLaserOdometry)
        {

            //　数据接收标志位清空
            newLaserCloudCornerLast = false; 
            newLaserCloudSurfLast = false;  
            newLaserOdometry = false;

            std::lock_guard<std::mutex> lock(mtx);

            // 确保每次全局优化的时间时间间隔不能太快，大于0.3s
            if (timeLaserOdometry - timeLastProcessing >= mappingProcessInterval) {
                // 储存里程计数据时间戳
                timeLastProcessing = timeLaserOdometry;

                transformAssociateToMap();

                // 提取上一帧优化得到的位姿附近的临近点
                // 合并成局部地图
                extractSurroundingKeyFrames();

                // 当前帧的 边缘点集合，平面点集合降采样
                downsampleCurrentScan();

                // 当前扫描进行优化，图优化以及进行LM优化的过程
                scan2MapOptimization();

                // 关键帧判断，gtsam优化
                saveKeyFramesAndFactor();

                //
                // correctPoses();

                publishTF();

                publishKeyPosesAndFrames();

                clearCloud();
            }
        }
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "lego_loam");

    ROS_INFO("\033[1;32m---->\033[0m Map Optimization Started.");

    mapOptimization MO;

    // std::thread 构造函数，将MO作为参数传入构造的线程中使用
    // 回环线程
    // std::thread loopthread(&mapOptimization::loopClosureThread, &MO);
	
    // 该线程中进行的工作是publishGlobalMap(),将数据发布到ros中，可视化
    // std::thread visualizeMapThread(&mapOptimization::visualizeGlobalMapThread, &MO);

    ros::Rate rate(200);
    while (ros::ok())
    {
        ros::spinOnce();

        MO.run();

        rate.sleep();
    }

    // loopthread.join();
    // visualizeMapThread.join();

    return 0;
}