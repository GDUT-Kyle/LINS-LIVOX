#include <unordered_map>
#include "livox_ros_driver/CustomMsg.h"
#include "lins_livox/common.h"
#include "lins_livox/MapRingBuffer.h"
#include "lins_livox/FilterState.h"
#include "lins_livox/myImu.h"

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>

#include <pcl/filters/radius_outlier_removal.h>   //半径滤波器头文件

#define USE_COMPLEMENTLY_FILTER

class fusionOdometry
{
private:
    ros::NodeHandle nh;
    ros::Subscriber subLaserCloud;
    ros::Subscriber subLaserCloudInfo;
    ros::Subscriber subGroundCloud;
    ros::Subscriber subImu;
    ros::Subscriber subPlaneEquation;

    ros::Publisher pubGroundDS;
    ros::Publisher pubCornerPointsSharp;
    ros::Publisher pubCornerPointsLessSharp;
    ros::Publisher pubSurfPointsFlat;
    ros::Publisher pubSurfPointsLessFlat;
    ros::Publisher pubLaserCloudCornerLast;
    ros::Publisher pubLaserCloudSurfLast;
    ros::Publisher pubLaserOdometry;

    std_msgs::Header cloudHeader;
    std_msgs::Header cloudHeaderLast;

    MapRingBuffer<pcl::PointCloud<PointType>> pclBuf;
    MapRingBuffer<cloud_msgs::cloud_info> pclInfoBuf;
    MapRingBuffer<myImu> imuBuf;
    MapRingBuffer<pcl::PointCloud<PointType>> pclGndBuf;
    MapRingBuffer<Eigen::Matrix<float, 4, 1>> planeBuf;

    cloud_msgs::cloud_info segmented_cloud_info;
    pcl::PointCloud<PointType>::Ptr segmented_cloud;
    pcl::PointCloud<PointType>::Ptr ground_cloud;
    Eigen::Matrix<float, 4, 1> planeEqu;
    Eigen::Matrix<float, 4, 1> planeEquLast;

    pcl::PointCloud<PointType>::Ptr surfPointsGroundScan;
    pcl::PointCloud<PointType>::Ptr surfPointsGroundScanDS;
    pcl::PointCloud<PointType>::Ptr surfPointsGroundScanLast;
    pcl::VoxelGrid<PointType> GoundDownSizeFilter;

    enum FusionState{
        STATUS_INIT = 0,
        STATUS_FIRST_SCAN = 1,
        STATUS_SECOND_SCAN = 2,
        STATUS_RUNNING = 3,
        STATUS_RESET = 4
    }status_;

    double pclTime, pclInfoTime, pclGndTime, planeTime;

    std::vector<smoothness_t> cloudSmoothness;
    float cloudCurvature[N_SCAN*Horizon_SCAN];
    int cloudNeighborPicked[N_SCAN*Horizon_SCAN];
    int cloudLabel[N_SCAN*Horizon_SCAN];

    pcl::PointCloud<PointType>::Ptr cornerPointsSharp;
    pcl::PointCloud<PointType>::Ptr cornerPointsLessSharp;
    pcl::PointCloud<PointType>::Ptr surfPointsFlat;
    pcl::PointCloud<PointType>::Ptr surfPointsLessFlat;

    pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScan;
    pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScanDS;

    pcl::VoxelGrid<PointType> downSizeFilter;

    pcl::PointCloud<PointType>::Ptr laserCloudCornerLast;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLast;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerLast;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfLast;

    int laserCloudCornerLastNum;
    int laserCloudSurfLastNum;

    FilterState preTransformCur;
    FilterState preTransformCurImu;
    float preTransformCurrRx;
    float preTransformCurrRy;
    float preTransformCurrRz;
    float preTransformCurTx;
    float preTransformCurTy;
    float preTransformCurTz;
    float preTransformCurVx;
    float preTransformCurVy;
    float preTransformCurVz;

    FilterState TransformSum;
    FilterState TransformSumLast;

    std::vector<myImu> ImuBucket;
    Eigen::Quaternionf initRotatin;
    bool initialImu;

    PointType pointOri, pointSel, tripod1, tripod2, tripod3, pointProj, coeff;
    std::vector<int> pointSearchInd;
    std::vector<float> pointSearchSqDis;

    int pointSelCornerInd[N_SCAN*Horizon_SCAN];
    std::unordered_map<int, std::pair<Eigen::Vector3f, Eigen::Vector3f>> pointSearchCornerInd;

    pcl::PointCloud<PointType>::Ptr laserCloudOri;
    pcl::PointCloud<PointType>::Ptr coeffSel;

    bool isDegenerate;
    Eigen::Matrix<float, 3, 3> matP;

    nav_msgs::Odometry laserOdometry;
    tf::TransformBroadcaster tfBroadcaster;
    tf::StampedTransform laserOdometryTrans;

    int skipFrameNum;
    int frameCount;

    Eigen::Quaternionf initImuMountAngle;

    Eigen::Matrix<float, 18, 18> F_t;
    Eigen::Matrix<float, 18, 12> G_t;
    Eigen::Matrix<float, 18, 18> P_t;
    Eigen::Matrix<float, 12, 12> noise_;
    float residual_;
    Eigen::Matrix<float, 1, 18> H_k;
    float R_k;
    Eigen::Matrix<float, 18, 1> K_k;
    Eigen::Matrix<float, 18, 1> updateVec_;
    Eigen::Matrix<float, 18, 1> errState;

    // ug: micro-gravity force -- 9.81/(10^6)
    double peba = pow(ACC_N * ug, 2);
    double pebg = pow(GYR_N * dph, 2);
    double pweba = pow(ACC_W * ugpsHz, 2);
    double pwebg = pow(GYR_W * dpsh, 2);
    Eigen::Vector3f gra_cov;

public:
    // 构造函数
    fusionOdometry(): nh("~")
    {
        // 订阅和发布各类话题
        subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>("/segmented_cloud", 1, &fusionOdometry::laserCloudHandler, this);
        subLaserCloudInfo = nh.subscribe<cloud_msgs::cloud_info>("/segmented_cloud_info", 1, &fusionOdometry::laserCloudInfoHandler, this);
        subGroundCloud = nh.subscribe<sensor_msgs::PointCloud2>("/ground_cloud", 1, &fusionOdometry::GroundCloudHandler, this);
        subImu = nh.subscribe<sensor_msgs::Imu>(imuTopic, 500, &fusionOdometry::imuHandler, this);
        subPlaneEquation = nh.subscribe<std_msgs::Float64MultiArray>("/plane_equation", 1, &fusionOdometry::PlaneEqHandler, this);

        pubGroundDS = nh.advertise<sensor_msgs::PointCloud2>("/ground_downsize", 2);
        pubCornerPointsSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_sharp", 1);
        pubCornerPointsLessSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp", 1);
        pubSurfPointsFlat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_flat", 1);
        pubSurfPointsLessFlat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_flat", 1);
        pubLaserCloudCornerLast = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_corner_last", 2);
        pubLaserCloudSurfLast = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 2);
        pubLaserOdometry = nh.advertise<nav_msgs::Odometry> ("/laser_odom_to_init", 5);

        initializationValue();
    }
    // 开辟内存函数
    void initializationValue()
    {
        segmented_cloud.reset(new pcl::PointCloud<PointType>());
        ground_cloud.reset(new pcl::PointCloud<PointType>());

        pclBuf.allocate(3);
        pclInfoBuf.allocate(3);
        pclGndBuf.allocate(3);
        planeBuf.allocate(3);
        imuBuf.allocate(500);

        status_ = STATUS_FIRST_SCAN;

        pclTime = 0.0; pclInfoTime = 0.0; pclGndTime = 0.0; planeTime = 0.0;

        surfPointsGroundScan.reset(new pcl::PointCloud<PointType>());
        surfPointsGroundScanDS.reset(new pcl::PointCloud<PointType>());
        surfPointsGroundScanLast.reset(new pcl::PointCloud<PointType>());
        GoundDownSizeFilter.setLeafSize(1.0, 1.0, 1.0);

        cloudSmoothness.resize(N_SCAN*Horizon_SCAN);

        cornerPointsSharp.reset(new pcl::PointCloud<PointType>());
        cornerPointsLessSharp.reset(new pcl::PointCloud<PointType>());
        surfPointsFlat.reset(new pcl::PointCloud<PointType>());
        surfPointsLessFlat.reset(new pcl::PointCloud<PointType>());

        surfPointsLessFlatScan.reset(new pcl::PointCloud<PointType>());
        surfPointsLessFlatScanDS.reset(new pcl::PointCloud<PointType>());

        downSizeFilter.setLeafSize(0.2, 0.2, 0.2);

        laserCloudCornerLast.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfLast.reset(new pcl::PointCloud<PointType>());

        kdtreeCornerLast.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeSurfLast.reset(new pcl::KdTreeFLANN<PointType>());

        preTransformCur.setIdentity();
        preTransformCurImu.setIdentity();
        TransformSum.setIdentity();
        TransformSumLast.setIdentity();

        initRotatin.setIdentity();
        initialImu = false;

        initImuMountAngle.setIdentity();

        laserCloudOri.reset(new pcl::PointCloud<PointType>());
        coeffSel.reset(new pcl::PointCloud<PointType>());

        isDegenerate = false;
        matP = Eigen::Matrix<float, 3, 3>::Zero();

        laserOdometry.header.frame_id = "/camera_init";
        laserOdometry.child_frame_id = "/camera_odom";

        laserOdometryTrans.frame_id_ = "/camera_init";
        laserOdometryTrans.child_frame_id_ = "/camera_odom";

        skipFrameNum = 1;
        frameCount = skipFrameNum;

        preTransformCurrRx = 0.0;
        preTransformCurrRy = 0.0;
        preTransformCurrRz = 0.0;
        preTransformCurTx = 0.0;
        preTransformCurTy = 0.0;
        preTransformCurTz = 0.0;
        preTransformCurVx = 0.0;
        preTransformCurVy = 0.0;
        preTransformCurVz = 0.0;

        F_t.setZero();
        G_t.setZero();
        P_t.setZero();
        noise_.setZero();
        // asDiagonal()指将向量作为对角线构建对角矩阵
        noise_.block<3, 3>(0, 0) = Eigen::Vector3f(peba, peba, peba).asDiagonal();
        noise_.block<3, 3>(3, 3) = Eigen::Vector3f(pebg, pebg, pebg).asDiagonal();
        noise_.block<3, 3>(6, 6) = Eigen::Vector3f(pweba, pweba, pweba).asDiagonal();
        noise_.block<3, 3>(9, 9) = Eigen::Vector3f(pwebg, pwebg, pwebg).asDiagonal();
        residual_ = 0.0;
        H_k.setZero();
        R_k = 0.0;
        K_k.setZero();
        updateVec_.setZero();
        errState.setZero();
        gra_cov << 0.01, 0.01, 0.01;

        #ifndef INITIAL_BY_IMU
        Eigen::AngleAxisf imuPitch = Eigen::AngleAxisf(livox_mount_pitch, Eigen::Vector3f::UnitY());
        Eigen::AngleAxisf imuRoll = Eigen::AngleAxisf(livox_mount_roll, Eigen::Vector3f::UnitX());
        initImuMountAngle = imuRoll * imuPitch;
        #endif
    }
    // 析构函数
    ~fusionOdometry(){}
    // 订阅segmented_cloud
    void laserCloudHandler(const sensor_msgs::PointCloud2ConstPtr& msg)
    {
        segmented_cloud->clear();
        pcl::fromROSMsg(*msg, *segmented_cloud);
        pclBuf.addMeas(*segmented_cloud, msg->header.stamp.toSec());
    }
    // 订阅segmented_cloud_info
    void laserCloudInfoHandler(const cloud_msgs::cloud_infoConstPtr& msg)
    {
        pclInfoBuf.addMeas(*msg, msg->header.stamp.toSec());
    }
    // 订阅ground_cloud
    void GroundCloudHandler(const sensor_msgs::PointCloud2ConstPtr& msg)
    {
        ground_cloud->clear();
        pcl::fromROSMsg(*msg, *ground_cloud);
        pclGndBuf.addMeas(*ground_cloud, msg->header.stamp.toSec());
    }
    // 订阅plane_equation
    void PlaneEqHandler(const std_msgs::Float64MultiArrayConstPtr& planeIn)
    {
        Eigen::Matrix<float, 4, 1> planeEqu_;
        planeEqu_(0, 0) = planeIn->data[1];
        planeEqu_(1, 0) = planeIn->data[2];
        planeEqu_(2, 0) = planeIn->data[3];
        planeEqu_(3, 0) = planeIn->data[4];
        planeBuf.addMeas(planeEqu_, planeIn->data[0]);
    }
    // 订阅IMU
    void imuHandler(const sensor_msgs::Imu::ConstPtr& imuIn)
    {
        imuBuf.addMeas(myImu(imuIn), imuIn->header.stamp.toSec());
    }

    void alignFirstIMUtoPCL()
    {
        imuBuf.clean(pclInfoTime);
    }

    void DownSizeGroudCloud()
    {
        *surfPointsGroundScan = *ground_cloud;
        // 进行一次强降采样作为本轮特征点，同时去除离群点
        GoundDownSizeFilter.setLeafSize(0.5, 0.5, 0.5);
        GoundDownSizeFilter.setInputCloud(surfPointsGroundScan);
        GoundDownSizeFilter.filter(*surfPointsGroundScanDS);
        pcl::RadiusOutlierRemoval<PointType> radiusoutlier;  //创建滤波器
        radiusoutlier.setInputCloud(surfPointsGroundScanDS);    //设置输入点云
        radiusoutlier.setRadiusSearch(5);     //设置半径为100的范围内找临近点
        radiusoutlier.setMinNeighborsInRadius(30); //设置查询点的邻域点集数小于2的删除
        radiusoutlier.filter(*surfPointsGroundScanDS);

        // 进行一次弱降采样用于下一轮的最近邻搜索，并且直接存储在surfPointsGroundScan
        GoundDownSizeFilter.setLeafSize(0.3, 0.3, 0.3);
        GoundDownSizeFilter.setInputCloud(surfPointsGroundScan);
        GoundDownSizeFilter.filter(*surfPointsGroundScan);

        if(pubGroundDS.getNumSubscribers() != 0)
        {
            sensor_msgs::PointCloud2 msg_tmp;
            pcl::toROSMsg(*surfPointsGroundScanDS, msg_tmp);
            msg_tmp.header.stamp = cloudHeader.stamp;
            msg_tmp.header.frame_id = cloudHeader.frame_id;
            pubGroundDS.publish(msg_tmp);
        }
    }

    void calculateSmoothness()
    {
        constexpr float kDistanceFaraway = 25;
        int kNumCurvSize = 3;
        int cloudSize = segmented_cloud->points.size();

        for(int i=5; i<cloudSize-5; i++)
        {
            // 计算点的探测深度
            float dis = segmented_cloud_info.segmentedCloudRange[i];
            // 太远?
            if (dis > kDistanceFaraway)
                kNumCurvSize = 2;
            else
                kNumCurvSize = 3;
            
            float diffRange = 0.0;
            for(int j=kNumCurvSize; j!=0; j--)
            {
                diffRange += (segmented_cloud_info.segmentedCloudRange[i-j]
                            + segmented_cloud_info.segmentedCloudRange[i+j]);
            }
            diffRange -= (2 * kNumCurvSize * segmented_cloud_info.segmentedCloudRange[i]);

            cloudCurvature[i] = diffRange * diffRange;

            // 在markOccludedPoints()函数中对该参数进行重新修改
            cloudNeighborPicked[i] = 0;
            // 在extractFeatures()函数中会对标签进行修改，
			// 初始化为0，surfPointsFlat标记为-1，surfPointsLessFlatScan为不大于0的标签
			// cornerPointsSharp标记为2，cornerPointsLessSharp标记为1
            cloudLabel[i] = 0;

            cloudSmoothness[i].value = cloudCurvature[i] / (2 * kNumCurvSize * dis + 1e-3);
            cloudSmoothness[i].ind = i;
        }
    }

    void markOccludedPoints()
    {
        int cloudSize = segmented_cloud->points.size();

        for(int i=5; i<cloudSize-6; i++)
        {
            // 地面点暂时不管
            if(segmented_cloud_info.segmentedCloudGroundFlag[i] == 1)
                continue;
            float depth1 = segmented_cloud_info.segmentedCloudRange[i];
            float depth2 = segmented_cloud_info.segmentedCloudRange[i+1];
            int columnDiff = std::abs(int(segmented_cloud_info.segmentedCloudColInd[i+1]-segmented_cloud_info.segmentedCloudColInd[i]));

            if(columnDiff < 6)
            {
                // 选择距离较远的那些点，并将他们标记为1(不可选为特征点)
                if (depth1 - depth2 > 0.2){
                    // cloudNeighborPicked[i - 5] = 1;
                    // cloudNeighborPicked[i - 4] = 1;
                    cloudNeighborPicked[i - 3] = 1;
                    cloudNeighborPicked[i - 2] = 1;
                    cloudNeighborPicked[i - 1] = 1;
                    cloudNeighborPicked[i] = 1;
                }else if (depth2 - depth1 > 0.2){
                    cloudNeighborPicked[i + 1] = 1;
                    cloudNeighborPicked[i + 2] = 1;
                    cloudNeighborPicked[i + 3] = 1;
                    // cloudNeighborPicked[i + 4] = 1;
                    // cloudNeighborPicked[i + 5] = 1;
                    // cloudNeighborPicked[i + 6] = 1;
                }
            }
            float diff1 = std::abs(segmented_cloud_info.segmentedCloudRange[i-1] - segmented_cloud_info.segmentedCloudRange[i]);
            float diff2 = std::abs(segmented_cloud_info.segmentedCloudRange[i+1] - segmented_cloud_info.segmentedCloudRange[i]);

            // 选择距离变化较大的点，并将他们标记为1
            if(diff1>0.02*segmented_cloud_info.segmentedCloudRange[i] && diff2>0.02*segmented_cloud_info.segmentedCloudRange[i])
                cloudNeighborPicked[i] = 1;
        }
    }

    void extractFeatures()
    {
        cornerPointsSharp->clear();
        cornerPointsLessSharp->clear();
        surfPointsFlat->clear();
        surfPointsLessFlat->clear();

        // 按线遍历
        for(int i=0; i<N_SCAN; i++)
        {
            surfPointsLessFlatScan->clear();

            // 将每条scan平均分50块
            for(int j=0; j<50; j++)
            {
                // 每一块的首尾索引,sp和ep的含义:startPointer,endPointer
                int sp = segmented_cloud_info.startRingIndex[i] + 
                        (segmented_cloud_info.endRingIndex[i] - segmented_cloud_info.startRingIndex[i]) * j/50;
                int ep = segmented_cloud_info.startRingIndex[i] + 
                        (segmented_cloud_info.endRingIndex[i] - segmented_cloud_info.startRingIndex[i]) * (j+1)/50 - 1;

                if(sp >= ep)
                    continue;

                // 按照cloudSmoothness.value从小到大排序
                std::sort(cloudSmoothness.begin()+sp, cloudSmoothness.begin()+ep, by_value());

                int largestPickedNum = 0;
                float SumCurRegion = 0.0;
                for(int k=ep-1; k>=sp; k--)
                {
                    SumCurRegion += cloudCurvature[k];
                }
                for(int k=ep; k>=sp; k--)
                {
                    // 因为上面对cloudSmoothness进行了一次从小到大排序，所以ind不一定等于k了
                    int ind = cloudSmoothness[k].ind; // 获取最大曲率的那个索引
                    // 还未被选取、曲率大于阈值、不是地面点的话，则该点将被选取为特征点
                    if(cloudNeighborPicked[ind]==0 &&
                        cloudCurvature[ind] > edgeThreshold &&
                        // cloudCurvature[ind] > SumCurRegion && // 判断这个点的曲率是否足够突出,不然不能当做特征点
                        segmented_cloud_info.segmentedCloudGroundFlag[ind] == false)
                    {
                        largestPickedNum++;
                        if(largestPickedNum<=2)
                        {
                            // 论文中nFe=2,cloudSmoothness已经按照从小到大的顺序排列，
                            // 所以这边只要选择最后两个放进队列即可
                            // cornerPointsSharp标记为2
                            cloudLabel[ind] = 2;
                            cornerPointsSharp->push_back(segmented_cloud->points[ind]);
                            cornerPointsLessSharp->push_back(segmented_cloud->points[ind]);
                        }
                        else if(largestPickedNum<=20)
                        {
                            // 塞20个点到cornerPointsLessSharp中去
							// cornerPointsLessSharp标记为1
                            cloudLabel[ind] = 1;
                            cornerPointsLessSharp->push_back(segmented_cloud->points[ind]);
                        }
                        else
                            break;

                        // 看看特征点附近是否有断点
                        cloudNeighborPicked[ind] = 1;
                        for(int l=1; l<=3; l++)
                        {
                            // 从ind+l开始后面3个点，每个点index之间的差值，
                            // 确保columnDiff<=6,然后标记为我们需要的点
                            int columnDiff = std::abs(int(segmented_cloud_info.segmentedCloudRange[ind+l] - segmented_cloud_info.segmentedCloudRange[ind+l-1]));
                            if(columnDiff > 6)
                                break;
                            cloudNeighborPicked[ind+l] = 1;
                        }
                        for(int l=-1; l>=-3; l--)
                        {
                            // 从ind+l开始前面五个点，计算差值然后标记
                            int columnDiff = std::abs(int(segmented_cloud_info.segmentedCloudColInd[ind + l] - segmented_cloud_info.segmentedCloudColInd[ind + l + 1]));
                            if (columnDiff > 6)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                    SumCurRegion -= cloudCurvature[ind];
                }

                // 从地面点中筛选surfPointsFlat
                int smallestPickedNum = 0;
                for(int k=sp; k<=ep; k++)
                {
                    int ind = cloudSmoothness[k].ind;
                    // 平面点只从地面点中进行选择? 
                    if(cloudNeighborPicked[ind]==0 &&
                        cloudCurvature[ind]<surfThreshold &&
                        // segInfo.segmentedCloudRange[ind] > 10 &&
                        segmented_cloud_info.segmentedCloudGroundFlag[ind]==true)
                    {
                        cloudLabel[ind] = -1;
                        // surfPointsFlat->push_back(segmented_cloud->points[ind]);

                        // 论文中nFp=4，将4个最平的平面点放入队列中
                        smallestPickedNum++;
                        if(smallestPickedNum>=4)
                        {
                            break;
                        }

                        cloudNeighborPicked[ind] = 1;
                        for(int l=1; l<=5; l++)
                        {
                            // 从前面往后判断是否是需要的邻接点，是的话就进行标记
                            int columnDiff = std::abs(int(segmented_cloud_info.segmentedCloudColInd[ind+l]-segmented_cloud_info.segmentedCloudColInd[ind+l-1]));
                            if(columnDiff > 10)
                            {
                                break;
                            }
                            cloudNeighborPicked[ind+l] = 1;
                        }
                        for(int l=-1; l>=-5; l--)
                        {
                            // 从后往前开始标记 
                            int columnDiff = std::abs(int(segmented_cloud_info.segmentedCloudColInd[ind + l] - segmented_cloud_info.segmentedCloudColInd[ind + l + 1]));
                            if (columnDiff > 10)
                                break;

                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }

                // 非特征点的话，都归surfPointsLessFlatScan
                for(int k=sp; k<=ep; k++)
                {
                    if(cloudLabel[k] <= 0)
                    {
                        surfPointsLessFlatScan->push_back(segmented_cloud->points[k]);
                    }
                }
            }
            // surfPointsLessFlatScan中有过多的点云，如果点云太多，计算量太大
            // 进行下采样，可以大大减少计算量
            surfPointsLessFlatScanDS->clear();
            downSizeFilter.setInputCloud(surfPointsLessFlatScan);
            downSizeFilter.filter(*surfPointsLessFlatScanDS);

            *surfPointsLessFlat += *surfPointsLessFlatScanDS;
        }

        surfPointsFlat->clear();
        *surfPointsFlat = *surfPointsGroundScanDS;
    }

    void publishCloud()
    {
        sensor_msgs::PointCloud2 laserCloudOutMsg;

	    if (pubCornerPointsSharp.getNumSubscribers() != 0){
	        pcl::toROSMsg(*cornerPointsSharp, laserCloudOutMsg);
	        laserCloudOutMsg.header.stamp = cloudHeader.stamp;
	        laserCloudOutMsg.header.frame_id = cloudHeader.frame_id;
	        pubCornerPointsSharp.publish(laserCloudOutMsg);
	    }

	    if (pubCornerPointsLessSharp.getNumSubscribers() != 0){
	        pcl::toROSMsg(*cornerPointsLessSharp, laserCloudOutMsg);
	        laserCloudOutMsg.header.stamp = cloudHeader.stamp;
	        laserCloudOutMsg.header.frame_id = cloudHeader.frame_id;
	        pubCornerPointsLessSharp.publish(laserCloudOutMsg);
	    }

	    if (pubSurfPointsFlat.getNumSubscribers() != 0){
	        pcl::toROSMsg(*surfPointsFlat, laserCloudOutMsg);
	        laserCloudOutMsg.header.stamp = cloudHeader.stamp;
	        laserCloudOutMsg.header.frame_id = cloudHeader.frame_id;
	        pubSurfPointsFlat.publish(laserCloudOutMsg);
	    }

	    if (pubSurfPointsLessFlat.getNumSubscribers() != 0){
	        pcl::toROSMsg(*surfPointsLessFlat, laserCloudOutMsg);
	        laserCloudOutMsg.header.stamp = cloudHeader.stamp;
	        laserCloudOutMsg.header.frame_id = cloudHeader.frame_id;
	        pubSurfPointsLessFlat.publish(laserCloudOutMsg);
	    }
    }

    void checkSystemInitialization(){
        // 交换cornerPointsLessSharp和laserCloudCornerLast
        pcl::PointCloud<PointType>::Ptr laserCloudTemp = cornerPointsLessSharp;
        cornerPointsLessSharp = laserCloudCornerLast;
        laserCloudCornerLast = laserCloudTemp;

        // 交换surfPointsLessFlat和laserCloudSurfLast
        laserCloudTemp = surfPointsLessFlat;
        surfPointsLessFlat = laserCloudSurfLast;
        laserCloudSurfLast = laserCloudTemp;

        *surfPointsGroundScanLast = *surfPointsGroundScan;

        // 用于下一次搜索角点
        kdtreeCornerLast->setInputCloud(laserCloudCornerLast);
        // 用于下一次计算pitch, roll, tz
        planeEquLast = planeEqu;

        laserCloudCornerLastNum = laserCloudCornerLast->points.size();
        laserCloudSurfLastNum = surfPointsGroundScanLast->points.size();

        sensor_msgs::PointCloud2 laserCloudCornerLast2;
        pcl::toROSMsg(*laserCloudCornerLast, laserCloudCornerLast2);
        laserCloudCornerLast2.header.stamp = cloudHeader.stamp;
        laserCloudCornerLast2.header.frame_id = cloudHeader.frame_id;
        pubLaserCloudCornerLast.publish(laserCloudCornerLast2);

        sensor_msgs::PointCloud2 laserCloudSurfLast2;
        pcl::toROSMsg(*laserCloudSurfLast, laserCloudSurfLast2);
        laserCloudSurfLast2.header.stamp = cloudHeader.stamp;
        laserCloudSurfLast2.header.frame_id = cloudHeader.frame_id;
        pubLaserCloudSurfLast.publish(laserCloudSurfLast2);

        status_ = STATUS_SECOND_SCAN;

        cloudHeaderLast = cloudHeader;
        TransformSumLast = TransformSum;
    }

    void getImuBucket(double pcl_time)
    {
        ImuBucket.clear();
        double imuTime = 0.0;
        myImu imuTmp;
        imuBuf.getFirstTime(imuTime);
        #ifdef INITIAL_BY_IMU
        // 其实这里的IMU初始姿态选取有点粗暴，直接取了启动时第一个IMU数据计算姿态
        if(!initialImu)
        {
            imuBuf.getFirstMeas(imuTmp);
            // float imupitch_ = atan2(-imuTmp.acc.x(), 
            //                   sqrt(imuTmp.acc.z()*imuTmp.acc.z() + 
            //                   imuTmp.acc.y()*imuTmp.acc.y()));
            // float imuroll_ = atan2(imuTmp.acc.y(), imuTmp.acc.z());
            int signAccZ;
            if(imuTmp.acc.z() >= 0) signAccZ = 1;
            else signAccZ = -1;
            float imupitch_ = -signAccZ * asin(imuTmp.acc.x()/gnorm);
            float imuroll_ = signAccZ * asin(imuTmp.acc.y()/gnorm);
            Eigen::AngleAxisf imuPitch = Eigen::AngleAxisf(imupitch_, Eigen::Vector3f::UnitY());
            Eigen::AngleAxisf imuRoll = Eigen::AngleAxisf(imuroll_, Eigen::Vector3f::UnitX());
            initImuMountAngle = imuRoll * imuPitch;
            initialImu = true;
        }
        #endif
        while(imuTime <= pcl_time && !imuBuf.empty())
        {
            imuBuf.getFirstMeas(imuTmp);
            ImuBucket.emplace_back(imuTmp);
            imuBuf.clean(imuTime);
            imuBuf.getFirstTime(imuTime);
        }
        // 再从队列中取一个新的Imu测量用于中值积分
        if(!imuBuf.empty()) 
        {
            imuBuf.getFirstMeas(imuTmp);
            ImuBucket.emplace_back(imuTmp);
        }
    }

    void prediction()
    {
        getImuBucket(pclInfoTime);
        // std::cout<<"ImuBucket : "<<ImuBucket.size()<<std::endl;
        double imuTime_last = ImuBucket[0].timestamp_s;
        double dt = 0.0;
        FilterState state_tmp = TransformSum;
        // 加速度转到世界系
        // ImuBucket[0].acc = Ext_Livox.rotation() * ImuBucket[0].acc;
        Eigen::Vector3f un_acc_last = state_tmp.qbn_*initImuMountAngle*(ImuBucket[0].acc-state_tmp.ba_)+state_tmp.gn_;
        Eigen::Vector3f un_gyr_last = ImuBucket[0].gyr - state_tmp.bw_;
        Eigen::Vector3f un_acc_next;
        Eigen::Vector3f un_gyr_next;
        for(int i=1; i<ImuBucket.size(); i++)
        {
            dt = ImuBucket[i].timestamp_s - imuTime_last;
            // 加速度转到世界系
            // ImuBucket[i].acc = Ext_Livox.rotation() * ImuBucket[i].acc;
            un_acc_next = state_tmp.qbn_*initImuMountAngle*(ImuBucket[i].acc-state_tmp.ba_)+state_tmp.gn_;
            un_gyr_next = ImuBucket[i].gyr - state_tmp.bw_;

            Eigen::Vector3f un_acc = 0.5*(un_acc_last + un_acc_next); // world frame
            Eigen::Vector3f un_gyr = 0.5*(un_gyr_last + un_gyr_next);
            // 求角度变化量，再转化成四元数形式
            Eigen::Quaternionf dq = axis2Quat(un_gyr * dt);
            // 求当前临时状态量中的姿态
            state_tmp.qbn_ = (state_tmp.qbn_*dq).normalized();

            state_tmp.vn_ = state_tmp.vn_ + un_acc*dt; // world frame
            state_tmp.rn_ = state_tmp.rn_ + state_tmp.vn_*dt + un_acc*dt*dt; // world frame

            imuTime_last = ImuBucket[i].timestamp_s;
            un_acc_last = un_acc_next;
            un_gyr_last = un_gyr_next;
        }

        preTransformCurImu.rn_ = state_tmp.qbn_.inverse()*(state_tmp.rn_-TransformSumLast.rn_);
        preTransformCurImu.vn_ = state_tmp.qbn_.inverse()*state_tmp.vn_;
        preTransformCurImu.qbn_ = (state_tmp.qbn_ * TransformSumLast.qbn_.inverse()).normalized();
        preTransformCur = preTransformCurImu;
        updateTransformCurTmp();
        updateTransformCur();
    }

    void TransformToStart(PointType const * const pi, PointType * const po)
    {
        // float s = 10 * (pi->intensity - int(pi->intensity));
        // std::cout<<"s :"<<s<<std::endl;
        float s = 1.0;
        Eigen::Vector3f point(pi->x, pi->y, pi->z);
        point = Eigen::Quaternionf::Identity().slerp(s, preTransformCur.qbn_) 
                * point + preTransformCur.rn_;
        po->x = point.x();
        po->y = point.y();
        po->z = point.z();
        po->intensity = pi->intensity;
        po->curvature = pi->curvature;
    }

    void findCorrespondingCornerFeatures(int iterCount){
        int cornerPointsSharpNum = cornerPointsSharp->points.size();
        // 遍历当前帧的角点
        for(int i=0; i<cornerPointsSharpNum; i++)
        {
            TransformToStart(&cornerPointsSharp->points[i], &pointSel);
            // pointSel = cornerPointsSharp->points[i];

            // 每5次迭代更新一次匹配特征
            if(iterCount % 5 == 0)
            {
                // 从上一帧中寻找最近邻点
                kdtreeCornerLast->radiusSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);
                // 确保搜索点足够近
                if(pointSearchSqDis[4] < DISTANCE_SQ_THRESHOLD)
                {
                    std::vector<Eigen::Vector3f> nearCorners;
                    Eigen::Vector3f center(0.0, 0.0, 0.0);
                    for(int j=0; j<5; j++)
                    {
                        Eigen::Vector3f tmp(
                            laserCloudCornerLast->points[pointSearchInd[j]].x,
                            laserCloudCornerLast->points[pointSearchInd[j]].y,
                            laserCloudCornerLast->points[pointSearchInd[j]].z
                        );
                        center = center + tmp;
                        nearCorners.push_back(tmp);
                    }
                    center = center / 5.0;

                    Eigen::Matrix3f covMat = Eigen::Matrix3f::Zero();
                    for(int j=0; j<5; j++)
                    {
                        Eigen::Vector3f tmpZeroMean = 
                            nearCorners[j] - center;
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
                        Eigen::Vector3f last_point_a, last_point_b;
                        last_point_a = 0.1 * unit_direction + point_on_line;
                        last_point_b = -0.1 * unit_direction + point_on_line; 
                        pointSearchCornerInd[i] = std::pair<Eigen::Vector3f, Eigen::Vector3f>(last_point_a, last_point_b);
                    }
                }
            }
            // 确保存在特征匹配对
            if(pointSearchCornerInd.find(i) != pointSearchCornerInd.end())
            {
                //选择的特征点记为O，kd-tree最近距离点记为A，另一个最近距离点记为B
                Eigen::Vector3f tmp_a;
                Eigen::Vector3f tmp_b;
                tmp_a = pointSearchCornerInd[i].first;
                tmp_b = pointSearchCornerInd[i].second;

                Eigen::Vector3f tmp_c = 0.5*(tmp_a + tmp_b);
                Eigen::Vector3f tmp_p(pointSel.x, pointSel.y, pointSel.z);
                // 计算点线距离
                Eigen::Matrix<float, 3, 1> nu = (tmp_p - tmp_a).cross(tmp_p - tmp_b); //(叉乘)
                Eigen::Matrix<float, 3, 1> de = tmp_a - tmp_b;
                Eigen::Matrix<float, 3, 1> ld = de.cross(nu);

                // if(ld.dot(tmp_a - tmp_p)>0)
                //     ld = -ld;
                ld.normalize();

                float la = ld.x();
                float lb = ld.y();
                float lc = ld.z();
                float ld2 = nu.norm() / de.norm();

                //权重计算，距离越大权重越小，距离越小权重越大，得到的权重范围<=1
                float s = 1;
                if (iterCount >= 0) { //5次迭代之后开始增加权重因素
                    //增加权重，距离越远，影响影子越小
                    s = 1 - 1.8 * fabs(ld2);
                }
                // if(fabs(ld2) > 1.8)
                //     s = 0.0;

                if (s > 0.1 && ld2 != 0) {
                    coeff.x = s * la;
                    coeff.y = s * lb;
                    coeff.z = s * lc;
                    coeff.intensity = s * ld2;
                    
                    // 当前帧的特征点
                    laserCloudOri->push_back(cornerPointsSharp->points[i]);
                    // 特征点指到特征直线的垂线方向，其中intensity存储了点到直线的距离，也就是残差
                    coeffSel->push_back(coeff);
                }
            }
        }
    }

    void updateTransformCurTmp()
    {
        preTransformCurTx = preTransformCur.rn_.x();
        preTransformCurTy = preTransformCur.rn_.y();
        preTransformCurTz = preTransformCur.rn_.z();

        Eigen::AngleAxisf Angle(preTransformCur.qbn_.toRotationMatrix());
        preTransformCurrRx = Angle.angle() * Angle.axis().x();
        preTransformCurrRy = Angle.angle() * Angle.axis().y();
        preTransformCurrRz = Angle.angle() * Angle.axis().z();

        preTransformCurVx = preTransformCur.vn_.x();
        preTransformCurVy = preTransformCur.vn_.y();
        preTransformCurVz = preTransformCur.vn_.z();
        // std::cout<<"rotation : ("<<preTransformCurrRx<<", "<<preTransformCurrRy<<", "<<preTransformCurrRz<<")"<<std::endl;
    }

    void updateTransformCur()
    {
        preTransformCur.rn_ =  Eigen::Vector3f(preTransformCurTx, preTransformCurTy, preTransformCurTz);
        Eigen::Vector3f axisAngle(preTransformCurrRx, preTransformCurrRy, preTransformCurrRz);
        Eigen::AngleAxisf Angle = Eigen::AngleAxisf(axisAngle.norm(), axisAngle.normalized());
        preTransformCur.qbn_ = Angle;
        preTransformCur.qbn_.normalize();
        preTransformCur.updateVelocity(float(cloudHeader.stamp.toSec()-cloudHeaderLast.stamp.toSec()));
    }

    bool calculateTransformationCorner(int iterCount){
        int pointSelNum = laserCloudOri->points.size();

        Eigen::Vector3f v_pointOri_bk1;

        Eigen::Matrix<float, 1, 3> j_n;
        Eigen::Matrix3f H = Eigen::Matrix3f::Zero();
        Eigen::Vector3f b = Eigen::Vector3f::Zero();

        Eigen::Vector3f Delta_x = Eigen::Vector3f::Zero();

        for(int i=0; i<pointSelNum; i++)
        {
            // 当前点，在b_k+1坐标系
            pointOri = laserCloudOri->points[i];
            // 由当前点指向直线特征的垂线方向向量，其中intensity为距离值
            coeff = coeffSel->points[i];

            // 1. 计算G函数，通过估计的变换transformCur将pointOri转换到b_k坐标系
            v_pointOri_bk1.x() = pointOri.x;
            v_pointOri_bk1.y() = pointOri.y;
            v_pointOri_bk1.z() = pointOri.z;
            // v_pointOri_bk = q_transformCur.inverse() * v_pointOri_bk1 - v_transformCur;
            
            // 2. dD/dG = (la, lb, lc)
            Eigen::Matrix<float, 1, 3> dDdG;
            dDdG << coeff.x, coeff.y, coeff.z;
            // 3. 将transformCur转成R，然后计算(-Rp)^
            Eigen::Matrix3f neg_Rp_sym;
            // anti_symmetric(q_transformCur.toRotationMatrix()*v_pointOri_bk1, neg_Rp_sym);
            // neg_Rp_sym = -neg_Rp_sym;
            anti_symmetric(v_pointOri_bk1, neg_Rp_sym);
            neg_Rp_sym = -preTransformCur.qbn_.toRotationMatrix()*neg_Rp_sym;
            // 4. 计算(dD/dG)*(-Rp)^得到关于旋转的雅克比，取其中的yaw部分，记为j_yaw
            Eigen::Matrix<float, 1, 3> dDdR = dDdG * neg_Rp_sym;
            // 5. 计算关于平移的雅克比，即为(dD/dG)，取其中的x,y部分，记为j_x,j_y
            // 6. 组织该点的雅克比：[j_yaw,j_x,j_y]
            j_n(0, 0) = dDdR(0, 2);
            j_n(0, 1) = dDdG(0, 0);
            j_n(0, 2) = dDdG(0, 1);
            
            // 7. 该点的残差值,f(x)=coeff.intensity
            float f_n = 0.05 * coeff.intensity;
            // 8. 组织近似海森矩阵H += J^T*J
            H = H + j_n.transpose() * j_n;
            // 9. 组织方程右边：b += -J^T*f(x)
            b = b - f_n * j_n.transpose();
            // 10. 计算总残差用于判断收敛性cost += f(x)*f(x)

            // Delta_x -= (j_n.transpose()/pointSelNum);
        }
        // 11. 解方程：H*dx = b
        Delta_x = H.colPivHouseholderQr().solve(b);

        if(iterCount == 0)
        {
            Eigen::Matrix<float, 1, 3> matE = Eigen::Matrix<float, 1, 3>::Zero();
            Eigen::Matrix<float, 3, 3> matV = Eigen::Matrix<float, 3, 3>::Zero();
            Eigen::Matrix<float, 3, 3> matV2 = Eigen::Matrix<float, 3, 3>::Zero();

            // 计算At*A的特征值和特征向量
            // 特征值存放在matE，特征向量matV
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigenSolver(H);
            matE = eigenSolver.eigenvalues();
            matV = eigenSolver.eigenvectors();

            matV2 = matV;
            // 退化的具体表现是指什么？
            isDegenerate = false;
            float eignThre[3] = {10, 10, 10};
            for (int i = 2; i >= 0; i--) {
                if (matE(0, i) < eignThre[i]) {
                    for (int j = 0; j < 3; j++) {
                        matV2(i, j) = 0;
                    }
                    // 存在比10小的特征值则出现退化
                    isDegenerate = true;
                } else {
                    break;
                }
            }
            matP = matV.inverse() * matV2;
            // std::cout<<iterCount<<"] isDegenerate : "<<isDegenerate<<std::endl;
        }

        if (isDegenerate) {
            Eigen::Matrix<float, 3, 1> matX2 = Eigen::Matrix<float, 3, 1>::Zero();
            // matX.copyTo(matX2);
            matX2 = Delta_x;
            Delta_x = matP * matX2;
        }

        preTransformCurrRz += Delta_x[0];
        preTransformCurTx += Delta_x[1];
        preTransformCurTy += Delta_x[2];
        updateTransformCur();

        float deltaR = sqrt(pow(rad2deg(Delta_x(0, 0)), 2));
        float deltaT = sqrt( pow(Delta_x(1, 0) * 100, 2) +
                                pow(Delta_x(2, 0) * 100, 2));

        if (deltaR < 0.1 && deltaT < 0.1) {
            return false;
        }

        return true; 
    }

    void calculateTransformationGround()
    {
        Eigen::Vector3f NormalCurr = planeEqu.block<3, 1>(0, 0);
        float dLast = planeEqu(3, 0);
        float pitchCurr = atan2(NormalCurr.x(), sqrt(1.0-NormalCurr.x()*NormalCurr.x()));
        float rollCurr = atan2(NormalCurr.y(), NormalCurr.z());

        Eigen::Vector3f NormalLast = planeEquLast.block<3, 1>(0, 0);
        float dCurr = planeEquLast(3, 0);
        float pitchLast = atan2(NormalLast.x(), sqrt(1.0-NormalLast.x()*NormalLast.x()));
        float rollLast = atan2(NormalLast.y(), NormalLast.z());

        // 更新
        float droll, dpitch, dtz;
        droll = rollCurr - rollLast;
        dpitch = pitchCurr - pitchLast;
        dtz = dCurr-dLast;

        preTransformCurrRx = (-droll+preTransformCurrRx)/2;
        preTransformCurrRy = (-dpitch+preTransformCurrRy)/2;
        preTransformCurTz = (-dtz+preTransformCurTz)/2;
        updateTransformCur();
        
        planeEquLast = planeEqu;
    }

    void calculateTransformationGround_filter()
    {
        Eigen::Vector3f NormalCurr = planeEqu.block<3, 1>(0, 0);
        float dLast = planeEqu(3, 0);
        float pitchCurr = atan2(NormalCurr.x(), sqrt(1.0-NormalCurr.x()*NormalCurr.x()));
        float rollCurr = atan2(NormalCurr.y(), NormalCurr.z());

        Eigen::Vector3f NormalLast = planeEquLast.block<3, 1>(0, 0);
        float dCurr = planeEquLast(3, 0);
        float pitchLast = atan2(NormalLast.x(), sqrt(1.0-NormalLast.x()*NormalLast.x()));
        float rollLast = atan2(NormalLast.y(), NormalLast.z());

        // 更新
        float droll, dpitch, dtz;
        droll = rollCurr - rollLast;
        dpitch = pitchCurr - pitchLast;
        dtz = dCurr-dLast;

        preTransformCurrRx = -droll;
        preTransformCurrRy = -dpitch;
        preTransformCurTz = -dtz;
        updateTransformCur();

        // std::cout<<"roll = "<<droll/PI*180<<", pitch = "<<dpitch/PI*180<<", tz = "<<dtz<<std::endl;

        planeEquLast = planeEqu;
    }
    
    void updateTransformation(){
        // std::cout<<laserCloudSurfLastNum<<std::endl;
        if (laserCloudCornerLastNum < 10 || laserCloudSurfLastNum < 100)
            return;

        // 既然我们在预处理那花了这么多心思提取地平面，那我们索性直接使用地平面的法向量变化计算roll, pitch, 和z
        calculateTransformationGround();

        // std::cout<<"transform : "<< transformCur[0]<<", "<<transformCur[2]<<", "<<transformCur[4]<<std::endl;
        // std::cout<<"surf coeffSel : "<<coeffSel->size();
        pointSearchCornerInd.clear();
        // lastCost = 0.0;
        for (int iterCount2 = 0; iterCount2 < 25; iterCount2++) {

            laserCloudOri->clear();
            coeffSel->clear();

            // 找到对应的特征边/角点
            // 寻找边特征的方法和寻找平面特征的很类似，过程可以参照寻找平面特征的注释
            findCorrespondingCornerFeatures(iterCount2);

            if (laserCloudOri->points.size() < 10)
                continue;
            // 通过角/边特征的匹配，计算变换矩阵
            if (calculateTransformationCorner(iterCount2) == false)
                break;
        }
        // std::cout<<"velocity1 : "<<preTransformCur.vn_.transpose()<<std::endl;
        // std::cout<<"velocity2 : "<<preTransformCurImu.vn_.transpose()<<std::endl;
        // std::cout<<"-----------------------------------------"<<std::endl;
        // 互补滤波
        // #define USE_COMPLEMENTLY_FILTER
        // #ifdef USE_COMPLEMENTLY_FILTER
        // float s = 0.5;
        // preTransformCur.rn_ = s*preTransformCur.rn_+(1-s)*preTransformCurImu.rn_;
        // preTransformCur.vn_ = s*preTransformCur.vn_+(1-s)*preTransformCurImu.vn_;
        // preTransformCur.qbn_ = preTransformCur.qbn_.slerp(0.5, preTransformCurImu.qbn_);
        // #endif
    }

    bool calculateTransformationCorner_ieskf(int iterCount)
    {
        int pointSelNum = laserCloudOri->points.size();

        double residualNorm = 1e6;
        bool hasConverged = false;
        bool hasDiverged = false;

        float f_cost = 0.0;

        Eigen::Vector3f v_pointOri_bk1;
        H_k.setZero();

        #ifdef UNUSE_IESKF
        Eigen::Matrix<float, 18, 18> H;
        H.setZero();
        Eigen::Matrix<float, 18, 1> b;
        b.setZero();
        #endif

        for(int i=0; i<pointSelNum; i++)
        {
            // 当前点，在b_k+1坐标系
            pointOri = laserCloudOri->points[i];
            // 由当前点指向直线特征的垂线方向向量，其中intensity为距离值
            coeff = coeffSel->points[i];
            // 1. 计算G函数，通过估计的变换transformCur将pointOri转换到b_k坐标系
            v_pointOri_bk1.x() = pointOri.x;
            v_pointOri_bk1.y() = pointOri.y;
            v_pointOri_bk1.z() = pointOri.z;
            // v_pointOri_bk = q_transformCur.inverse() * v_pointOri_bk1 - v_transformCur;
            // 2. dD/dG = (la, lb, lc)
            Eigen::Matrix<float, 1, 3> dDdG;
            dDdG << coeff.x, coeff.y, coeff.z;
            // 3. 将transformCur转成R，然后计算(-Rp)^
            Eigen::Matrix3f neg_Rp_sym;
            // anti_symmetric(q_transformCur.toRotationMatrix()*v_pointOri_bk1, neg_Rp_sym);
            // neg_Rp_sym = -neg_Rp_sym;
            anti_symmetric(v_pointOri_bk1, neg_Rp_sym);
            neg_Rp_sym = -preTransformCur.qbn_.toRotationMatrix()*neg_Rp_sym;
            // 4. 计算(dD/dG)*(-Rp)^得到关于旋转的雅克比，取其中的yaw部分，记为j_yaw
            Eigen::Matrix<float, 1, 3> dDdR = dDdG * neg_Rp_sym;
            
            #ifdef UNUSE_IESKF
            H_k.block<1, 3>(0, pos_) = dDdG;
            H_k.block<1, 3>(0, att_) = dDdR;

            f_cost = 0.05 * coeff.intensity;

            b = b - f_cost*H_k.transpose();
            H = H + H_k.transpose()*H_k;
            #else
            H_k.block<1, 3>(0, pos_) += dDdG;
            H_k.block<1, 3>(0, att_) += dDdR;

            f_cost += 0.02 * coeff.intensity;
            #endif
        }

        #ifdef UNUSE_IESKF
        updateVec_ = -K_k*(f_cost-H_k*errState)-errState;
        updateVec_ = H.colPivHouseholderQr().solve(b);
        #else
        K_k = P_t*H_k.transpose()*(1/(H_k*P_t*H_k.transpose()+LIDAR_STD));
        updateVec_ = K_k*(H_k*errState - f_cost)-errState;
        #endif

        // Divergence determination
        // 迭代发散判断
        bool hasNaN = false;
        for (int i = 0; i < updateVec_.size(); i++) {
            if (isnan(updateVec_[i])) {
                updateVec_[i] = 0;
                hasNaN = true;
            }
        }
        if (hasNaN == true) {
            ROS_WARN("System diverges Because of NaN...");
            hasDiverged = true;
            return false;
        }
        // Check whether the filter converges
        // 检查滤波器是否迭代收敛
        if (f_cost > residualNorm * 10) {
            ROS_WARN("System diverges...");
            hasDiverged = true;
            return false;
        }

        errState += updateVec_;
        preTransformCurrRz += updateVec_(att_+2, 0);
        preTransformCurTx += updateVec_(pos_, 0);
        preTransformCurTy += updateVec_(pos_+1, 0);
        updateTransformCur();

        float deltaR = sqrt(pow(rad2deg(updateVec_(att_+2, 0)), 2));
        float deltaT = sqrt( pow(updateVec_(pos_, 0) * 100, 2) +
                                pow(updateVec_(pos_+1, 0) * 100, 2));

        if (deltaR < 0.1 && deltaT < 0.1) {
            return false;
        }

        return true; 
    }

    void updateTransformation_ieskf()
    {
        if (laserCloudCornerLastNum < 10 || laserCloudSurfLastNum < 100)
            return;

        // 互补滤波，计算roll, pitch, 和z并且更新
        calculateTransformationGround_filter();

        pointSearchCornerInd.clear();
        errState.setZero();
        for (int iterCount2 = 0; iterCount2 < 25; iterCount2++) {
            laserCloudOri->clear();
            coeffSel->clear();

            // 计算dD/dG
            findCorrespondingCornerFeatures(iterCount2);

            if (laserCloudOri->points.size() < 10)
                continue;
            // 通过角/边特征的匹配，计算变换矩阵
            if (calculateTransformationCorner_ieskf(iterCount2) == false)
                break;
        }
        P_t = (Eigen::Matrix<float, 18, 18>::Identity()-K_k*H_k)*P_t*(Eigen::Matrix<float, 18, 18>::Identity()-K_k*H_k).transpose()+
                R_k*K_k*K_k.transpose();
    }

    // 旋转角的累计变化量
    void integrateTransformation(){
        // 存储odom坐标系下的当前位置
        TransformSum.rn_ = TransformSum.rn_ + TransformSum.qbn_ * preTransformCur.rn_;
        TransformSum.qbn_ = TransformSum.qbn_ * preTransformCur.qbn_;
        TransformSum.vn_ = TransformSum.qbn_ * preTransformCur.vn_;

        TransformSumLast = TransformSum;
    }

    void publishOdometry(){
        // rx,ry,rz转化为四元数发布
        laserOdometry.header.stamp = cloudHeader.stamp;
        laserOdometry.pose.pose.orientation.x = TransformSum.qbn_.x();
        laserOdometry.pose.pose.orientation.y = TransformSum.qbn_.y();
        laserOdometry.pose.pose.orientation.z = TransformSum.qbn_.z();
        laserOdometry.pose.pose.orientation.w = TransformSum.qbn_.w();
        laserOdometry.pose.pose.position.x = TransformSum.rn_.x();
        laserOdometry.pose.pose.position.y = TransformSum.rn_.y();
        laserOdometry.pose.pose.position.z = TransformSum.rn_.z();
        pubLaserOdometry.publish(laserOdometry);

        // laserOdometryTrans 是用于tf广播
        laserOdometryTrans.stamp_ = cloudHeader.stamp;
        laserOdometryTrans.setRotation(tf::Quaternion(TransformSum.qbn_.x(), TransformSum.qbn_.y(), TransformSum.qbn_.z(), TransformSum.qbn_.w()));
        laserOdometryTrans.setOrigin(tf::Vector3(TransformSum.rn_.x(), TransformSum.rn_.y(), TransformSum.rn_.z()));
        tfBroadcaster.sendTransform(laserOdometryTrans);
    }

    void publishCloudsLast(){
        pcl::PointCloud<PointType>::Ptr laserCloudTemp = cornerPointsLessSharp;
        // 最新的cornerPointsLessSharp已经发布出去了，所以它已经没意义了，不过为了节省空间，就让暂存一下数据
        cornerPointsLessSharp = laserCloudCornerLast;
        // 用于下一轮的kdtree搜索，注意是less sharp点
        laserCloudCornerLast = laserCloudTemp;

        laserCloudTemp = surfPointsLessFlat;
        surfPointsLessFlat = laserCloudSurfLast;
        laserCloudSurfLast = laserCloudTemp;

        laserCloudCornerLastNum = laserCloudCornerLast->points.size();
        // laserCloudSurfLastNum = laserCloudSurfLast->points.size();
        laserCloudSurfLastNum = surfPointsGroundScan->points.size();

        if(laserCloudCornerLastNum>10 && laserCloudSurfLastNum>100)
        {
            kdtreeCornerLast->setInputCloud(laserCloudCornerLast);
            #ifdef PLANE_EQU
            // kdtreeSurfLast->setInputCloud(laserCloudSurfLast);
            kdtreeSurfLast->setInputCloud(surfPointsGroundScan);
            #endif
        }

        surfPointsGroundScanLast->clear();
        *surfPointsGroundScanLast = *surfPointsGroundScan;

        frameCount++;

        if (frameCount >= skipFrameNum + 1) {

            frameCount = 0;

            sensor_msgs::PointCloud2 laserCloudCornerLast2;
            pcl::toROSMsg(*laserCloudCornerLast, laserCloudCornerLast2);
            laserCloudCornerLast2.header.stamp = cloudHeader.stamp;
            laserCloudCornerLast2.header.frame_id = "/livox";
            pubLaserCloudCornerLast.publish(laserCloudCornerLast2);

            sensor_msgs::PointCloud2 laserCloudSurfLast2;
            pcl::toROSMsg(*laserCloudSurfLast, laserCloudSurfLast2);
            laserCloudSurfLast2.header.stamp = cloudHeader.stamp;
            laserCloudSurfLast2.header.frame_id = "/livox";
            pubLaserCloudSurfLast.publish(laserCloudSurfLast2);
        }

        cloudHeaderLast = cloudHeader;
    }


    // 滤波器的预测
    void filter_predict()
    {
        getImuBucket(pclInfoTime);
        // std::cout<<"ImuBucket : "<<ImuBucket.size()<<std::endl;
        double imuTime_last = ImuBucket[0].timestamp_s;
        double dt = 0.0;
        FilterState state_tmp = TransformSum;
        // 加速度转到世界系
        // ImuBucket[0].acc = Ext_Livox.rotation() * ImuBucket[0].acc;
        Eigen::Vector3f un_acc_last = state_tmp.qbn_*initImuMountAngle*(ImuBucket[0].acc-state_tmp.ba_)+state_tmp.gn_;
        Eigen::Vector3f un_gyr_last = ImuBucket[0].gyr - state_tmp.bw_;
        Eigen::Vector3f un_acc_next;
        Eigen::Vector3f un_gyr_next;
        for(int i=1; i<ImuBucket.size(); i++)
        {
            dt = ImuBucket[i].timestamp_s - imuTime_last;
            // 加速度转到世界系
            // ImuBucket[i].acc = Ext_Livox.rotation() * ImuBucket[i].acc;
            un_acc_next = state_tmp.qbn_*initImuMountAngle*(ImuBucket[i].acc-state_tmp.ba_)+state_tmp.gn_;
            un_gyr_next = ImuBucket[i].gyr - state_tmp.bw_;

            Eigen::Vector3f un_acc = 0.5*(un_acc_last + un_acc_next); // world frame
            Eigen::Vector3f un_gyr = 0.5*(un_gyr_last + un_gyr_next);
            // 求角度变化量，再转化成四元数形式
            Eigen::Quaternionf dq = axis2Quat(un_gyr * dt);
            // 求当前临时状态量中的姿态
            state_tmp.qbn_ = (state_tmp.qbn_*dq).normalized();

            state_tmp.vn_ = state_tmp.vn_ + un_acc*dt; // world frame
            state_tmp.rn_ = state_tmp.rn_ + state_tmp.vn_*dt + un_acc*dt*dt; // world frame

            imuTime_last = ImuBucket[i].timestamp_s;
            un_acc_last = un_acc_next;
            un_gyr_last = un_gyr_next;

            preTransformCurImu.rn_ = state_tmp.qbn_.inverse()*(state_tmp.rn_-TransformSumLast.rn_);
            preTransformCurImu.vn_ = state_tmp.qbn_.inverse()*state_tmp.vn_;
            preTransformCurImu.qbn_ = (state_tmp.qbn_ * TransformSumLast.qbn_.inverse()).normalized();
            preTransformCur = preTransformCurImu;
            updateTransformCurTmp();
            updateTransformCur();

            F_t.setZero();
            F_t.block<3, 3>(pos_, vel_) = Eigen::Matrix<float, 3, 3>::Identity();
            F_t.block<3, 3>(vel_, att_) = -preTransformCurImu.qbn_.toRotationMatrix()*anti_symmetric(ImuBucket[i].acc-state_tmp.ba_);
            F_t.block<3, 3>(vel_, acc_) = -preTransformCurImu.qbn_.toRotationMatrix();
            F_t.block<3, 3>(vel_, gra_) = Eigen::Matrix<float, 3, 3>::Identity();
            F_t.block<3, 3>(att_, att_) = -anti_symmetric(ImuBucket[i].gyr - state_tmp.bw_);
            F_t.block<3, 3>(att_, gyr_) = -Eigen::Matrix<float, 3, 3>::Identity();

            G_t.setZero();
            G_t.block<3, 3>(vel_, 0) = -preTransformCurImu.qbn_.toRotationMatrix();
            G_t.block<3, 3>(att_, 3) = -Eigen::Matrix<float, 3, 3>::Identity();
            G_t.block<3, 3>(acc_, 6) = Eigen::Matrix<float, 3, 3>::Identity();
            G_t.block<3, 3>(gyr_, 9) = Eigen::Matrix<float, 3, 3>::Identity();

            Eigen::Matrix<float, 18, 18> F_;
            F_ = Eigen::Matrix<float, 18, 18>::Identity() + dt*F_t;
            errState = F_ * errState;
            P_t = F_ * P_t * F_.transpose() + (dt*G_t) * noise_ * (dt*G_t).transpose();
        }
    }

    void clearCloud()
    {
        segmented_cloud->clear();
        ground_cloud->clear();
        surfPointsGroundScan->clear();
        surfPointsGroundScanDS->clear();
    }

    void run()
    {
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        // 如果有新数据进来则执行，否则不执行任何操作
        if(pclBuf.getSize() != 0 && pclInfoBuf.getSize() != 0 &&
            pclGndBuf.getSize() != 0 && planeBuf.getSize() != 0 &&
            imuBuf.getSize() != 0)
        {
            pclBuf.getLastTime(pclTime);
            pclInfoBuf.getLastTime(pclInfoTime);
            pclGndBuf.getLastTime(pclGndTime);
            planeBuf.getLastTime(planeTime);
            if(pclTime != pclInfoTime || pclGndTime != pclInfoTime || planeTime != pclInfoTime)
            {
                return;
            }

            pclBuf.getLastMeas(*segmented_cloud);
            pclInfoBuf.getLastMeas(segmented_cloud_info);
            pclGndBuf.getLastMeas(*ground_cloud);
            planeBuf.getLastMeas(planeEqu);
            cloudHeader = segmented_cloud_info.header;
            
            switch (status_)
            {
            case STATUS_INIT:
                break;
            case STATUS_FIRST_SCAN:
            {
                alignFirstIMUtoPCL();
                DownSizeGroudCloud();
                calculateSmoothness();
                markOccludedPoints();
                extractFeatures();
                publishCloud();
                checkSystemInitialization();
                break;
            }
            case STATUS_SECOND_SCAN:
            {
                #ifdef USE_COMPLEMENTLY_FILTER
                prediction();
                #endif
                DownSizeGroudCloud();
                calculateSmoothness();
                markOccludedPoints();
                extractFeatures();
                publishCloud();
                updateTransformation();
                integrateTransformation();
                publishOdometry();
                publishCloudsLast();
                status_ = STATUS_RUNNING;
                break;
            }
            case STATUS_RUNNING:
                filter_predict();
                DownSizeGroudCloud();
                calculateSmoothness();
                markOccludedPoints();
                extractFeatures();
                publishCloud();
                updateTransformation_ieskf();
                integrateTransformation();
                publishOdometry();
                publishCloudsLast();
                break;
            case STATUS_RESET:
                break;
            default:
                break;
            }
            // 6. pop头数据，维护队列
            pclBuf.clean(pclInfoTime);
            pclInfoBuf.clean(pclInfoTime);
            pclGndBuf.clean(pclInfoTime);
            planeBuf.clean(pclInfoTime);
            imuBuf.clean(pclInfoTime);
        }
        else{
            return;
        }

        clearCloud();

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
        // std::cout << "solve time cost = " << time_used.count() << " seconds. " << std::endl;
        ROS_DEBUG("solve time cost = %f seconds.", time_used.count());
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "lins");

    ROS_INFO("\033[1;32m---->\033[0m Fusion Odometry Started.");

    fusionOdometry FO;

    ros::Rate rate(400);
    while (ros::ok())
    {
        FO.run();
        ros::spinOnce();

        rate.sleep();
    }
    
    ros::spin();
    return 0;
}