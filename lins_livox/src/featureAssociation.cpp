#include <unordered_map>
#include "livox_ros_driver/CustomMsg.h"
#include "lins_livox/common.h"

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>

#include <pcl/filters/radius_outlier_removal.h>   //半径滤波器头文件

// #include <pcl/registration/icp.h>
// #include <pcl/common/transforms.h>

class featureAssociation
{
private:
    ros::NodeHandle nh;

    ros::Subscriber subLaserCloud;
    ros::Subscriber subLaserCloudInfo;
    // ros::Subscriber subOutlierCloud;
    ros::Subscriber subGroundCloud;
    ros::Subscriber subImu;

    ros::Subscriber subPlaneEquation;

    ros::Publisher pubLaserCloud_line;
    ros::Publisher pubLaserCloud_line_point;
    pcl::PointCloud<PointType>::Ptr line_Cloud;
    pcl::PointCloud<PointType>::Ptr line_point_Cloud;

    ros::Publisher pubGroundDS;

    ros::Publisher pubCornerPointsSharp;
    ros::Publisher pubCornerPointsLessSharp;
    ros::Publisher pubSurfPointsFlat;
    ros::Publisher pubSurfPointsLessFlat;

    pcl::PointCloud<PointType>::Ptr segmentedCloud;
    pcl::PointCloud<PointType>::Ptr outlierCloud;

    pcl::PointCloud<PointType>::Ptr cornerPointsSharp;
    pcl::PointCloud<PointType>::Ptr cornerPointsLessSharp;
    pcl::PointCloud<PointType>::Ptr surfPointsFlat;
    pcl::PointCloud<PointType>::Ptr surfPointsLessFlat;

    pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScan;
    pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScanDS;

    pcl::PointCloud<PointType>::Ptr surfPointsGroundScan;
    pcl::PointCloud<PointType>::Ptr surfPointsGroundScanDS;
    pcl::PointCloud<PointType>::Ptr surfPointsGroundScanLast;
    pcl::VoxelGrid<PointType> GoundDownSizeFilter;

    pcl::VoxelGrid<PointType> downSizeFilter;

    double timeScanCur;
    double timeNewSegmentedCloud;
    double timeNewSegmentedCloudInfo;
    double timeNewGroundCloud;
    double timePlaneEqu;

    bool newSegmentedCloud;
    bool newSegmentedCloudInfo;
    bool newGroundCloud;
    bool newPlaneEqu;

    cloud_msgs::cloud_info segInfo;
    std_msgs::Header cloudHeader;

    int systemInitCount;
    bool systemInited;

    std::vector<smoothness_t> cloudSmoothness;
    float cloudCurvature[N_SCAN*Horizon_SCAN];
    int cloudNeighborPicked[N_SCAN*Horizon_SCAN];
    int cloudLabel[N_SCAN*Horizon_SCAN];

    int imuPointerFront;
    int imuPointerLast;
    int imuPointerLastIteration;

    float imuRollStart, imuPitchStart, imuYawStart;
    float cosImuRollStart, cosImuPitchStart, cosImuYawStart, sinImuRollStart, sinImuPitchStart, sinImuYawStart;
    float imuRollCur, imuPitchCur, imuYawCur;

    float imuVeloXStart, imuVeloYStart, imuVeloZStart;
    float imuShiftXStart, imuShiftYStart, imuShiftZStart;

    float imuVeloXCur, imuVeloYCur, imuVeloZCur;
    float imuShiftXCur, imuShiftYCur, imuShiftZCur;

    float imuShiftFromStartXCur, imuShiftFromStartYCur, imuShiftFromStartZCur;
    float imuVeloFromStartXCur, imuVeloFromStartYCur, imuVeloFromStartZCur;

    float imuAngularRotationXCur, imuAngularRotationYCur, imuAngularRotationZCur;
    float imuAngularRotationXLast, imuAngularRotationYLast, imuAngularRotationZLast;
    float imuAngularFromStartX, imuAngularFromStartY, imuAngularFromStartZ;

    double imuTime[imuQueLength];
    float imuRoll[imuQueLength];
    float imuPitch[imuQueLength];
    float imuYaw[imuQueLength];

    float imuAccX[imuQueLength];
    float imuAccY[imuQueLength];
    float imuAccZ[imuQueLength];

    float imuVeloX[imuQueLength];
    float imuVeloY[imuQueLength];
    float imuVeloZ[imuQueLength];

    float imuShiftX[imuQueLength];
    float imuShiftY[imuQueLength];
    float imuShiftZ[imuQueLength];

    float imuAngularVeloX[imuQueLength];
    float imuAngularVeloY[imuQueLength];
    float imuAngularVeloZ[imuQueLength];

    float imuAngularRotationX[imuQueLength];
    float imuAngularRotationY[imuQueLength];
    float imuAngularRotationZ[imuQueLength];



    ros::Publisher pubLaserCloudCornerLast;
    ros::Publisher pubLaserCloudSurfLast;
    ros::Publisher pubLaserOdometry;
    ros::Publisher pubOutlierCloudLast;

    int skipFrameNum;
    bool systemInitedLM;

    int laserCloudCornerLastNum;
    int laserCloudSurfLastNum;

    int pointSelCornerInd[N_SCAN*Horizon_SCAN];
    std::unordered_map<int, std::pair<Eigen::Vector3f, Eigen::Vector3f>> pointSearchCornerInd;

    int pointSelSurfInd[N_SCAN*Horizon_SCAN];
    int pointSearchSurfInd1[N_SCAN*Horizon_SCAN];
    int pointSearchSurfInd2[N_SCAN*Horizon_SCAN];
    int pointSearchSurfInd3[N_SCAN*Horizon_SCAN];

    // 在k坐标系下，第i+1帧点云到第i帧点云的变换
    float transformCur[6];
    // odom坐标系下的位姿，是通过帧间匹配积分得到的
    float transformSum[6];

    Eigen::Vector3f v_transformCur;
    Eigen::AngleAxisf rollAngle;
    Eigen::AngleAxisf pitchAngle;
    Eigen::AngleAxisf yawAngle;
    Eigen::Quaternionf q_transformCur;

    float imuRollLast, imuPitchLast, imuYawLast;
    float imuShiftFromStartX, imuShiftFromStartY, imuShiftFromStartZ;
    float imuVeloFromStartX, imuVeloFromStartY, imuVeloFromStartZ;

    pcl::PointCloud<PointType>::Ptr laserCloudCornerLast;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLast;
    pcl::PointCloud<PointType>::Ptr laserCloudOri;
    pcl::PointCloud<PointType>::Ptr coeffSel;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerLast;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfLast;

    std::vector<int> pointSearchInd;
    std::vector<float> pointSearchSqDis;

    PointType pointOri, pointSel, tripod1, tripod2, tripod3, pointProj, coeff;

    nav_msgs::Odometry laserOdometry;

    tf::TransformBroadcaster tfBroadcaster;
    tf::StampedTransform laserOdometryTrans;

    bool isDegenerate;
    Eigen::Matrix<float, 3, 3> matP;

    int frameCount;

    float lastCost = 0.0;

    Eigen::Matrix<float, 4, 1> PlaneEqu;
    Eigen::Matrix<float, 4, 1> PlaneEquLast;

public:
    featureAssociation():
        nh("~")
        {
        // 订阅和发布各类话题
        subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>("/segmented_cloud", 1, &featureAssociation::laserCloudHandler, this);
        subLaserCloudInfo = nh.subscribe<cloud_msgs::cloud_info>("/segmented_cloud_info", 1, &featureAssociation::laserCloudInfoHandler, this);
        // subOutlierCloud = nh.subscribe<sensor_msgs::PointCloud2>("/outlier_cloud", 1, &featureAssociation::outlierCloudHandler, this);
        subGroundCloud = nh.subscribe<sensor_msgs::PointCloud2>("/ground_cloud", 1, &featureAssociation::GroundCloudHandler, this);
        subImu = nh.subscribe<sensor_msgs::Imu>(imuTopic, 50, &featureAssociation::imuHandler, this);
        subPlaneEquation = nh.subscribe<std_msgs::Float64MultiArray>("/plane_equation", 1, &featureAssociation::PlaneEqHandler, this);

        pubCornerPointsSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_sharp", 1);
        pubCornerPointsLessSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp", 1);
        pubSurfPointsFlat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_flat", 1);
        pubSurfPointsLessFlat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_flat", 1);

        pubLaserCloudCornerLast = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_corner_last", 2);
        pubLaserCloudSurfLast = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 2);
        pubOutlierCloudLast = nh.advertise<sensor_msgs::PointCloud2>("/outlier_cloud_last", 2);
        pubLaserOdometry = nh.advertise<nav_msgs::Odometry> ("/laser_odom_to_init", 5);

        pubLaserCloud_line_point = nh.advertise<sensor_msgs::PointCloud2>("/line_point", 2);
        pubLaserCloud_line = nh.advertise<sensor_msgs::PointCloud2>("/line", 2);

        pubGroundDS = nh.advertise<sensor_msgs::PointCloud2>("/ground_downsize", 2);
        
        initializationValue();
    }

    // 各类参数的初始化
    void initializationValue()
    {
        cloudSmoothness.resize(N_SCAN*Horizon_SCAN);

        // 下采样滤波器设置叶子间距，就是格子之间的最小距离
        downSizeFilter.setLeafSize(0.2, 0.2, 0.2);

        GoundDownSizeFilter.setLeafSize(1.0, 1.0, 1.0);

        segmentedCloud.reset(new pcl::PointCloud<PointType>());
        outlierCloud.reset(new pcl::PointCloud<PointType>());

        line_point_Cloud.reset(new pcl::PointCloud<PointType>());
        line_Cloud.reset(new pcl::PointCloud<PointType>());

        cornerPointsSharp.reset(new pcl::PointCloud<PointType>());
        cornerPointsLessSharp.reset(new pcl::PointCloud<PointType>());
        surfPointsFlat.reset(new pcl::PointCloud<PointType>());
        surfPointsLessFlat.reset(new pcl::PointCloud<PointType>());

        surfPointsLessFlatScan.reset(new pcl::PointCloud<PointType>());
        surfPointsLessFlatScanDS.reset(new pcl::PointCloud<PointType>());

        surfPointsGroundScan.reset(new pcl::PointCloud<PointType>());
        surfPointsGroundScanDS.reset(new pcl::PointCloud<PointType>());
        surfPointsGroundScanLast.reset(new pcl::PointCloud<PointType>());

        timeScanCur = 0;
        timeNewSegmentedCloud = 0;
        timeNewSegmentedCloudInfo = 0;
        timeNewGroundCloud = 0;
        timePlaneEqu = 0;

        newSegmentedCloud = false;
        newSegmentedCloudInfo = false;
        newGroundCloud = false;
        newPlaneEqu = false;

        systemInitCount = 0;
        systemInited = false;

        imuPointerFront = 0;
        imuPointerLast = -1;
        imuPointerLastIteration = 0;

        imuRollStart = 0; imuPitchStart = 0; imuYawStart = 0;
        cosImuRollStart = 0; cosImuPitchStart = 0; cosImuYawStart = 0;
        sinImuRollStart = 0; sinImuPitchStart = 0; sinImuYawStart = 0;
        imuRollCur = 0; imuPitchCur = 0; imuYawCur = 0;

        imuVeloXStart = 0; imuVeloYStart = 0; imuVeloZStart = 0;
        imuShiftXStart = 0; imuShiftYStart = 0; imuShiftZStart = 0;

        imuVeloXCur = 0; imuVeloYCur = 0; imuVeloZCur = 0;
        imuShiftXCur = 0; imuShiftYCur = 0; imuShiftZCur = 0;

        imuShiftFromStartXCur = 0; imuShiftFromStartYCur = 0; imuShiftFromStartZCur = 0;
        imuVeloFromStartXCur = 0; imuVeloFromStartYCur = 0; imuVeloFromStartZCur = 0;

        imuAngularRotationXCur = 0; imuAngularRotationYCur = 0; imuAngularRotationZCur = 0;
        imuAngularRotationXLast = 0; imuAngularRotationYLast = 0; imuAngularRotationZLast = 0;
        imuAngularFromStartX = 0; imuAngularFromStartY = 0; imuAngularFromStartZ = 0;

        for (int i = 0; i < imuQueLength; ++i)
        {
            imuTime[i] = 0;
            imuRoll[i] = 0; imuPitch[i] = 0; imuYaw[i] = 0;
            imuAccX[i] = 0; imuAccY[i] = 0; imuAccZ[i] = 0;
            imuVeloX[i] = 0; imuVeloY[i] = 0; imuVeloZ[i] = 0;
            imuShiftX[i] = 0; imuShiftY[i] = 0; imuShiftZ[i] = 0;
            imuAngularVeloX[i] = 0; imuAngularVeloY[i] = 0; imuAngularVeloZ[i] = 0;
            imuAngularRotationX[i] = 0; imuAngularRotationY[i] = 0; imuAngularRotationZ[i] = 0;
        }


        skipFrameNum = 1;

        for (int i = 0; i < 6; ++i){
            transformCur[i] = 0;
            transformSum[i] = 0;
        }
        q_transformCur.setIdentity();
        v_transformCur.setZero();

        systemInitedLM = false;

        imuRollLast = 0; imuPitchLast = 0; imuYawLast = 0;
        imuShiftFromStartX = 0; imuShiftFromStartY = 0; imuShiftFromStartZ = 0;
        imuVeloFromStartX = 0; imuVeloFromStartY = 0; imuVeloFromStartZ = 0;

        laserCloudCornerLast.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfLast.reset(new pcl::PointCloud<PointType>());
        laserCloudOri.reset(new pcl::PointCloud<PointType>());
        coeffSel.reset(new pcl::PointCloud<PointType>());

        // pointSearchCornerInd1.reset(new pcl::PointCloud<PointType>());
        // pointSearchCornerInd2.reset(new pcl::PointCloud<PointType>());

        kdtreeCornerLast.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeSurfLast.reset(new pcl::KdTreeFLANN<PointType>());

        laserOdometry.header.frame_id = "/camera_init";
        laserOdometry.child_frame_id = "/camera_odom";

        laserOdometryTrans.frame_id_ = "/camera_init";
        laserOdometryTrans.child_frame_id_ = "/camera_odom";
        
        isDegenerate = false;
        matP = Eigen::Matrix<float, 3, 3>::Zero();

        frameCount = skipFrameNum;

        PlaneEqu.setZero();
        PlaneEquLast.setZero();
    }

    ~featureAssociation(){};

    void laserCloudHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg){

        cloudHeader = laserCloudMsg->header;

        timeScanCur = cloudHeader.stamp.toSec();
        timeNewSegmentedCloud = timeScanCur;

        segmentedCloud->clear();
        pcl::fromROSMsg(*laserCloudMsg, *segmentedCloud);

        newSegmentedCloud = true;
    }

    void GroundCloudHandler(const sensor_msgs::PointCloud2ConstPtr& msgIn){

        timeNewGroundCloud = msgIn->header.stamp.toSec();

        surfPointsGroundScan->clear();
        pcl::fromROSMsg(*msgIn, *surfPointsGroundScan);

        newGroundCloud = true;
    }

    void laserCloudInfoHandler(const cloud_msgs::cloud_infoConstPtr& msgIn)
    {
        timeNewSegmentedCloudInfo = msgIn->header.stamp.toSec();
        segInfo = *msgIn;
        newSegmentedCloudInfo = true;
    }

    void imuHandler(const sensor_msgs::Imu::ConstPtr& imuIn)
    {
    }

    void PlaneEqHandler(const std_msgs::Float64MultiArrayConstPtr& planeIn)
    {
        timePlaneEqu = planeIn->data[0];
        // std::cout<<"planeIn : "<<planeIn->data[3]<<", "<<planeIn->data[4]<<std::endl;
        PlaneEqu(0, 0) = planeIn->data[1];
        PlaneEqu(1, 0) = planeIn->data[2];
        PlaneEqu(2, 0) = planeIn->data[3];
        PlaneEqu(3, 0) = planeIn->data[4];
        // std::cout<<"PlaneEqu : "<<PlaneEqu.transpose()<<std::endl;
        newPlaneEqu = true;
    }

    // void adjustDistortion()
    // {
    //     // 遍历点云中每个点
    //     for (int i = 0; i < cloudSize; i++) {
    //     }
    // }

    void DownSizeGroudCloud()
    {
        GoundDownSizeFilter.setLeafSize(0.5, 0.5, 0.5);
        GoundDownSizeFilter.setInputCloud(surfPointsGroundScan);
        GoundDownSizeFilter.filter(*surfPointsGroundScanDS);
 
        pcl::RadiusOutlierRemoval<PointType> radiusoutlier;  //创建滤波器
        
        radiusoutlier.setInputCloud(surfPointsGroundScanDS);    //设置输入点云
        radiusoutlier.setRadiusSearch(5);     //设置半径为100的范围内找临近点
        radiusoutlier.setMinNeighborsInRadius(30); //设置查询点的邻域点集数小于2的删除
        radiusoutlier.filter(*surfPointsGroundScanDS);

        // std::cout<<"surfPointsGroundScanDS : "<<surfPointsGroundScanDS->size()<<std::endl;

        GoundDownSizeFilter.setLeafSize(0.3, 0.3, 0.3);
        GoundDownSizeFilter.setInputCloud(surfPointsGroundScan);
        GoundDownSizeFilter.filter(*surfPointsGroundScan);

        if(pubGroundDS.getNumSubscribers() != 0)
        {
            sensor_msgs::PointCloud2 msg_tmp;
            pcl::toROSMsg(*surfPointsGroundScan, msg_tmp);
            msg_tmp.header.stamp = cloudHeader.stamp;
            msg_tmp.header.frame_id = "/livox";
            pubGroundDS.publish(msg_tmp);
        }
    }

    void adjustDistortion()
    {
        int cloudSize = segmentedCloud->points.size();
        PointType point;
        for(int i=0; i<cloudSize; i++)
        {
            point = segmentedCloud->points[i];
            // 用 point.intensity 的小数部分存储了时间偏移比率
            // 用于时间线性插值计算相对的畸变
        }
    }

    void calculateSmoothness()
    {
        constexpr float kDistanceFaraway = 25;
        int kNumCurvSize = 3;
        int cloudSize = segmentedCloud->points.size();

        for(int i=5; i<cloudSize-5; i++)
        {
            // 计算点的探测深度
            float dis = segInfo.segmentedCloudRange[i];
            // 太远?
            if (dis > kDistanceFaraway)
                kNumCurvSize = 2;
            else
                kNumCurvSize = 3;
            
            float diffRange = 0.0;
            for(int j=kNumCurvSize; j!=0; j--)
            {
                diffRange += (segInfo.segmentedCloudRange[i-j]
                            + segInfo.segmentedCloudRange[i+j]);
            }
            diffRange -= (2 * kNumCurvSize * segInfo.segmentedCloudRange[i]);

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

    // 阻塞点是哪种点?
    // 阻塞点指点云之间相互遮挡，而且又靠得很近的点
    // (除了LOAM中描述的两类无效点,还有预处理中聚类分割产生的断点)
    void markOccludedPoints()
    {
        int cloudSize = segmentedCloud->points.size();

        for(int i=5; i<cloudSize-6; i++)
        {
            // 地面点暂时不管
            if(segInfo.segmentedCloudGroundFlag[i] == 1)
                continue;
            float depth1 = segInfo.segmentedCloudRange[i];
            float depth2 = segInfo.segmentedCloudRange[i+1];
            int columnDiff = std::abs(int(segInfo.segmentedCloudColInd[i+1]-segInfo.segmentedCloudColInd[i]));

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
            float diff1 = std::abs(segInfo.segmentedCloudRange[i-1] - segInfo.segmentedCloudRange[i]);
            float diff2 = std::abs(segInfo.segmentedCloudRange[i+1] - segInfo.segmentedCloudRange[i]);

            // 选择距离变化较大的点，并将他们标记为1
            if(diff1>0.02*segInfo.segmentedCloudRange[i] && diff2>0.02*segInfo.segmentedCloudRange[i])
                cloudNeighborPicked[i] = 1;
        }
    }

    void extractFeatures()
    {
        cornerPointsSharp->clear();
        cornerPointsLessSharp->clear();
        surfPointsFlat->clear();
        surfPointsLessFlat->clear();

        // surfPointsGroundScan->clear();
        // surfPointsGroundScanDS->clear();

        // 按线遍历
        for(int i=0; i<N_SCAN; i++)
        {
            surfPointsLessFlatScan->clear();

            // 将每条scan平均分50块
            for(int j=0; j<50; j++)
            {
                // 每一块的首尾索引,sp和ep的含义:startPointer,endPointer
                int sp = segInfo.startRingIndex[i] + 
                        (segInfo.endRingIndex[i] - segInfo.startRingIndex[i]) * j/50;
                int ep = segInfo.startRingIndex[i] + 
                        (segInfo.endRingIndex[i] - segInfo.startRingIndex[i]) * (j+1)/50 - 1;

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
                        segInfo.segmentedCloudGroundFlag[ind] == false)
                    {
                        largestPickedNum++;
                        if(largestPickedNum<=2)
                        {
                            // 论文中nFe=2,cloudSmoothness已经按照从小到大的顺序排列，
                            // 所以这边只要选择最后两个放进队列即可
                            // cornerPointsSharp标记为2
                            cloudLabel[ind] = 2;
                            cornerPointsSharp->push_back(segmentedCloud->points[ind]);
                            cornerPointsLessSharp->push_back(segmentedCloud->points[ind]);
                        }
                        else if(largestPickedNum<=20)
                        {
                            // 塞20个点到cornerPointsLessSharp中去
							// cornerPointsLessSharp标记为1
                            cloudLabel[ind] = 1;
                            cornerPointsLessSharp->push_back(segmentedCloud->points[ind]);
                        }
                        else
                            break;

                        // 看看特征点附近是否有断点
                        cloudNeighborPicked[ind] = 1;
                        for(int l=1; l<=3; l++)
                        {
                            // 从ind+l开始后面3个点，每个点index之间的差值，
                            // 确保columnDiff<=6,然后标记为我们需要的点
                            int columnDiff = std::abs(int(segInfo.segmentedCloudRange[ind+l] - segInfo.segmentedCloudRange[ind+l-1]));
                            if(columnDiff > 6)
                                break;
                            cloudNeighborPicked[ind+l] = 1;
                        }
                        for(int l=-1; l>=-3; l--)
                        {
                            // 从ind+l开始前面五个点，计算差值然后标记
                            int columnDiff = std::abs(int(segInfo.segmentedCloudColInd[ind + l] - segInfo.segmentedCloudColInd[ind + l + 1]));
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
                        segInfo.segmentedCloudGroundFlag[ind]==true)
                    {
                        cloudLabel[ind] = -1;
                        surfPointsFlat->push_back(segmentedCloud->points[ind]);

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
                            int columnDiff = std::abs(int(segInfo.segmentedCloudColInd[ind+l]-segInfo.segmentedCloudColInd[ind+l-1]));
                            if(columnDiff > 10)
                            {
                                break;
                            }
                            cloudNeighborPicked[ind+l] = 1;
                        }
                        for(int l=-1; l>=-5; l--)
                        {
                            // 从后往前开始标记 
                            int columnDiff = std::abs(int(segInfo.segmentedCloudColInd[ind + l] - segInfo.segmentedCloudColInd[ind + l + 1]));
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
                        surfPointsLessFlatScan->push_back(segmentedCloud->points[k]);
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

        #ifdef PLANE_EQU
        surfPointsFlat->clear();
        *surfPointsFlat = *surfPointsGroundScanDS;
        #endif
    }

    void publishCloud()
    {
        sensor_msgs::PointCloud2 laserCloudOutMsg;

	    if (pubCornerPointsSharp.getNumSubscribers() != 0){
	        pcl::toROSMsg(*cornerPointsSharp, laserCloudOutMsg);
	        laserCloudOutMsg.header.stamp = cloudHeader.stamp;
	        laserCloudOutMsg.header.frame_id = "/livox";
	        pubCornerPointsSharp.publish(laserCloudOutMsg);
	    }

	    if (pubCornerPointsLessSharp.getNumSubscribers() != 0){
	        pcl::toROSMsg(*cornerPointsLessSharp, laserCloudOutMsg);
	        laserCloudOutMsg.header.stamp = cloudHeader.stamp;
	        laserCloudOutMsg.header.frame_id = "/livox";
	        pubCornerPointsLessSharp.publish(laserCloudOutMsg);
	    }

	    if (pubSurfPointsFlat.getNumSubscribers() != 0){
	        pcl::toROSMsg(*surfPointsFlat, laserCloudOutMsg);
	        laserCloudOutMsg.header.stamp = cloudHeader.stamp;
	        laserCloudOutMsg.header.frame_id = "/livox";
	        pubSurfPointsFlat.publish(laserCloudOutMsg);
	    }

	    if (pubSurfPointsLessFlat.getNumSubscribers() != 0){
	        pcl::toROSMsg(*surfPointsLessFlat, laserCloudOutMsg);
	        laserCloudOutMsg.header.stamp = cloudHeader.stamp;
	        laserCloudOutMsg.header.frame_id = "/livox";
	        pubSurfPointsLessFlat.publish(laserCloudOutMsg);
	    }
    }

    void TransformToStart(PointType const * const pi, PointType * const po)
    {
        // float s = 10 * (pi->intensity - int(pi->intensity));
        float s = 1.0;
        Eigen::Vector3f point(pi->x, pi->y, pi->z);
        point = Eigen::Quaternionf::Identity().slerp(s, q_transformCur) 
                * point + v_transformCur;
        po->x = point.x();
        po->y = point.y();
        po->z = point.z();
        po->intensity = pi->intensity;
        po->curvature = pi->curvature;
    }

    // 先转到start，再从start旋转到end
    void TransformToEnd(PointType const * const pi, PointType * const po)
    {
        // 先转会start
        // undistort point first
        PointType un_point_tmp;
        TransformToStart(pi, &un_point_tmp);

        Eigen::Vector3f un_point(un_point_tmp.x, un_point_tmp.y, un_point_tmp.z);
        un_point = q_transformCur.inverse() * (un_point - v_transformCur);

        po->x = un_point.x();
        po->y = un_point.y();
        po->z = un_point.z();
        po->intensity = pi->intensity;
        po->curvature = pi->curvature;
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

        kdtreeCornerLast->setInputCloud(laserCloudCornerLast);
        #ifdef PLANE_EQU
        // kdtreeSurfLast->setInputCloud(laserCloudSurfLast);
        kdtreeSurfLast->setInputCloud(surfPointsGroundScanLast);
        #endif

        laserCloudCornerLastNum = laserCloudCornerLast->points.size();
        // laserCloudSurfLastNum = laserCloudSurfLast->points.size();
        laserCloudSurfLastNum = surfPointsGroundScanLast->points.size();

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

        // transformSum[0] += imuPitchStart;
        // transformSum[2] += imuRollStart;

        PlaneEquLast = PlaneEqu;

        systemInitedLM = true;
    }

    void updateInitialGuess(){
    }

    #ifdef PLANE_EQU
    void findCorrespondingSurfFeatures(int iterCount){
        int surfPointsFlatNum = surfPointsFlat->points.size();
        if(iterCount % 5 == 0)
        {
            line_Cloud->clear();
            line_point_Cloud->clear();
        }
        // std::cout<<surfPointsFlatNum<<std::endl;
        float maxDis = 0.0;
        for (int i = 0; i < surfPointsFlatNum; i++) {
            // 坐标变换到开始时刻，参数0是输入，参数1是输出,校正运动畸变
            TransformToStart(&surfPointsFlat->points[i], &pointSel);
            // pointSel = surfPointsFlat->points[i];

            if(iterCount%5 == 0)
            {
                // k点最近邻搜索，这里k=1
                kdtreeSurfLast->radiusSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);
                Eigen::Matrix<float, 5, 3> matA0;
                Eigen::Matrix<float, 5, 1> matB0 = 
                    -1 * Eigen::Matrix<float, 5, 1>::Ones();
                if(pointSearchSqDis[4] < DISTANCE_SQ_THRESHOLD)
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
                        matA0(j, 0) = surfPointsGroundScanLast->points[pointSearchInd[j]].x;
                        matA0(j, 1) = surfPointsGroundScanLast->points[pointSearchInd[j]].y;
                        matA0(j, 2) = surfPointsGroundScanLast->points[pointSearchInd[j]].z;
                    }
                    // 计算平面的法向量
                    Eigen::Vector3f norm = matA0.colPivHouseholderQr().solve(matB0);
                    float negative_OA_dot_norm = 1/norm.norm();
                    norm.normalize(); // 归一化

                    // 将5个近邻点代入方程，验证
                    bool planeValid = true;
                    for(int j=0; j<5; j++)
                    {
                        // if OX * n > 0.2, then plane is not fit well
                        float ox_n = fabs(norm(0)*surfPointsGroundScanLast->points[pointSearchInd[j]].x+
                                            norm(1)*surfPointsGroundScanLast->points[pointSearchInd[j]].y+
                                            norm(2)*surfPointsGroundScanLast->points[pointSearchInd[j]].z + 
                                            negative_OA_dot_norm);
                        if(ox_n > 0.02)
                        {
                            planeValid = false;
                            break;
                        }
                    }

                    if(planeValid)
                    {
                        pointSearchSurfInd1[i] = pointSearchInd[0];
                        pointSearchSurfInd2[i] = pointSearchInd[2];
                        pointSearchSurfInd3[i] = pointSearchInd[4];
                        if(iterCount % 5 == 0)
                        {pointSel.intensity = i;
                        line_point_Cloud->push_back(surfPointsFlat->points[i]);
                        line_Cloud->push_back(surfPointsGroundScanLast->points[pointSearchSurfInd1[i]]);
                        line_Cloud->push_back(surfPointsGroundScanLast->points[pointSearchSurfInd2[i]]);
                        line_Cloud->push_back(surfPointsGroundScanLast->points[pointSearchSurfInd3[i]]);}
                    }
                }
            }
            // 前后都能找到对应的最近点在给定范围之内
            // 那么就开始计算距离
            // [pa,pb,pc]是tripod1，tripod2，tripod3这3个点构成的一个平面的方向量，
            // ps是模长，它是三角形面积的2倍
            if (pointSearchSurfInd2[i] >= 0 && pointSearchSurfInd3[i] >= 0) {
                tripod1 = surfPointsGroundScanLast->points[pointSearchSurfInd1[i]];
                tripod2 = surfPointsGroundScanLast->points[pointSearchSurfInd2[i]];
                tripod3 = surfPointsGroundScanLast->points[pointSearchSurfInd3[i]];

                Eigen::Vector3f tmp_a(tripod1.x, tripod1.y, tripod1.z);
                Eigen::Vector3f tmp_b(tripod2.x, tripod2.y, tripod2.z);
                Eigen::Vector3f tmp_c(tripod3.x, tripod3.y, tripod3.z);

                Eigen::Vector3f tmp_p(pointSel.x, pointSel.y, pointSel.z);

                Eigen::Vector3f ld = (tmp_b - tmp_a).cross(tmp_c - tmp_a); // 平面法向量
                ld.normalize();
                // 确保方向向量是由面指向点
                // if(ld.dot(tmp_a - tmp_p)>0)
                //     ld = -ld;

                // 距离不要取绝对值
                float pd2 = ld.dot(tmp_p - tmp_a);
                // if(pd2 > 0.1 || pd2 < -0.1) continue;
                // maxDis += pd2;

                // std::cout<<"pd2 = "<<pd2<<", ";

                float pa = ld.x();
                float pb = ld.y();
                float pc = ld.z();

                // if(pd2 > (10 * maxDis) && i>0)
                //     break;
                // maxDis = std::max(maxDis, pd2);
                float s = 1;
                if (iterCount >= 0) {
                    // 加上影响因子
                    s = 1 - 0.9 * fabs(pd2);
                }

                if (s > 0.4 && pd2 != 0) {
                    // [x,y,z]是整个平面的单位法量
                    // intensity是平面外一点到该平面的距离
                    coeff.x = s * pa;
                    coeff.y = s * pb;
                    coeff.z = s * pc;
                    coeff.intensity = s * pd2;

                    // 未经变换的点放入laserCloudOri队列，距离，法向量值放入coeffSel
                    laserCloudOri->push_back(surfPointsFlat->points[i]);
                    coeffSel->push_back(coeff);
                }
            }
        }
        // std::cout<<"size : "<<laserCloudOri->size()<<std::endl;
    }
    #endif

    void findCorrespondingCornerFeatures(int iterCount){
        int cornerPointsSharpNum = cornerPointsSharp->points.size();
        // if(iterCount % 5 == 0)
        // {
        //     line_Cloud->clear();
        //     line_point_Cloud->clear();
        // }
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

                        // if(iterCount % 5 == 0)
                        // {pointSel.intensity = i;
                        // line_point_Cloud->push_back(cornerPointsSharp->points[i]);
                        // PointType point = pointSel;
                        // point.x = last_point_a.x();
                        // point.y = last_point_a.y();
                        // point.z = last_point_a.z();
                        // line_Cloud->push_back(point);
                        // point.x = last_point_b.x();
                        // point.y = last_point_b.y();
                        // point.z = last_point_b.z();
                        // line_Cloud->push_back(point);}
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

    void updateTransformCur()
    {
        v_transformCur =  Eigen::Vector3f(transformCur[3], transformCur[4], transformCur[5]);
        rollAngle = Eigen::AngleAxisf(transformCur[0], Eigen::Vector3f::UnitX());
        pitchAngle = Eigen::AngleAxisf(transformCur[1], Eigen::Vector3f::UnitY());
        yawAngle = Eigen::AngleAxisf(transformCur[2], Eigen::Vector3f::UnitZ());
        // Eigen::AngleAxisf Angle = Eigen::AngleAxisf(
        //     sqrt(transformCur[0]*transformCur[0] + 
        //         transformCur[1]*transformCur[1] + 
        //         transformCur[2]*transformCur[2])
        //     , Eigen::Vector3f(transformCur[0], transformCur[1], transformCur[2]));
        q_transformCur = rollAngle * pitchAngle * yawAngle;
        // q_transformCur = Angle;
    }

    #ifdef PLANE_EQU
    bool calculateTransformationSurf(int iterCount){

        {
        // // Eigen::Vector3f sum = Eigen::Vector3f::Zero();
        // // Eigen::Vector3f mean = Eigen::Vector3f::Zero();
        // // Eigen::Matrix3f cov = Eigen::Matrix3f::Zero();
        // // Eigen::Vector3f matE = Eigen::Vector3f::Zero();
        // // Eigen::Vector3f matV = Eigen::Vector3f::Zero();
        // // Eigen::Matrix3f H = Eigen::Matrix3f::Zero();
        // // Eigen::Vector3f b = Eigen::Vector3f::Zero();
        // Eigen::Matrix3f cov;
        // Eigen::Vector4f pc_mean1;
        // Eigen::Vector4f pc_mean2;
        // // 计算地面种子的协方差和均值
        // pcl::computeMeanAndCovarianceMatrix(*line_point_Cloud, cov, pc_mean1);
        // // Singular Value Decomposition: SVD
        // // 计算协方差矩阵的奇异值
        // Eigen::JacobiSVD<Eigen::MatrixXf> svd(cov,Eigen::DecompositionOptions::ComputeFullU);
        // // use the least singular vector as normal
        // // 挑选出最小的特征值对应的特征向量，这是地平面的法向量
        // Eigen::MatrixXf normal_1 = (svd.matrixU().col(2));
        // // if(svd.singularValues()[2] > 0.1 * svd.singularValues()[1])
        // // {
        // //     // std::cout<<"1matV : "<<normal_1.transpose()<<std::endl;
        // //     // std::cout<<"roll : "<<-atan2(normal_1(0, 1), 1)<<", pitch : "<<atan2(normal_1(0, 0), 1)<<std::endl;
        // //     return false;
        // // }

        // // 计算地面种子的协方差和均值
        // pcl::computeMeanAndCovarianceMatrix(*line_Cloud, cov, pc_mean2);
        // svd.compute(cov);
        // Eigen::MatrixXf normal_2 = (svd.matrixU().col(2));
        // // if(svd.singularValues()[2] > 0.1 * svd.singularValues()[1])
        // // {
        // //     // std::cout<<"2matV : "<<normal_2.transpose()<<std::endl;
        // //     // std::cout<<"roll : "<<-atan2(normal_2(0, 1), 1)<<", pitch : "<<atan2(normal_2(0, 0), 1)<<std::endl;
        // //     return false;
        // // }
        // transformCur[0] = -atan2(normal_1(1, 0), normal_1(2, 0)) - (-atan2(normal_2(1, 0), normal_2(2, 0)));
        // transformCur[1] = -(atan2(normal_1(0, 0), normal_1(2, 0)) - atan2(normal_2(0, 0), normal_2(2, 0)));
        // transformCur[5] = normal_2.block<1,3>(0,0).dot((pc_mean2-pc_mean1).head<3>());
        // updateTransformCur();

        // std::cout<<"transformCur[0] = "<<transformCur[0]<<", "<<"transformCur[1] = "<<transformCur[1]<<std::endl;
        // return true;
        }

        int pointSelNum = laserCloudOri->points.size();

        // std::cout<<"pointSelNum : "<<pointSelNum<<std::endl;

        Eigen::Vector3f v_pointOri_bk1;

        Eigen::Matrix3f H = Eigen::Matrix3f::Zero();
        Eigen::Vector3f b = Eigen::Vector3f::Zero();
        float cost = 0.0;

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
            neg_Rp_sym = -q_transformCur.toRotationMatrix()*neg_Rp_sym;
            // 4. 计算(dD/dG)*(-Rp)^得到关于旋转的雅克比，取其中的yaw部分，记为j_yaw
            Eigen::Matrix<float, 1, 3> dDdR = dDdG * neg_Rp_sym;
            // 5. 计算关于平移的雅克比，即为(dD/dG)，取其中的x,y部分，记为j_x,j_y
            // 6. 组织该点的雅克比：[j_yaw,j_x,j_y]
            Eigen::Matrix<float, 1, 3> j_n;
            j_n(0, 0) = dDdR(0, 0);
            j_n(0, 1) = dDdR(0, 1);
            j_n(0, 2) = dDdG(0, 2);
            
            // 7. 该点的残差值,f(x)=coeff.intensity
            float f_n = 0.05 * coeff.intensity;
            // 8. 组织近似海森矩阵H += J^T*J
            H = H + j_n.transpose() * j_n;
            // 9. 组织方程右边：b += -J^T*f(x)
            b = b - f_n * j_n.transpose();
            // 10. 计算总残差用于判断收敛性cost += f(x)*f(x)
            cost += f_n * f_n;

            // Delta_x += 0.01 * std::fabs(f_n)*(j_n.transpose());
        }
        // 11. 解方程：H*dx = b
        Delta_x = H.colPivHouseholderQr().solve(b);
        // 12. 首次迭代需要通过SVD分解H矩阵检测退化情况：如果
        // 如果是第一次求解
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
            // Delta_x.normalize();
        }

        transformCur[0] += Delta_x[0];
        transformCur[1] += Delta_x[1];
        transformCur[5] += Delta_x[2];

        // std::cout<<iterCount<<"] "<<"Delta_x : "<<Delta_x.transpose()<<std::endl;

        for(int i=0; i<6; i++){
            // std::cout<<transformCur[i]<<", ";
            if(isnan(transformCur[i]))
                transformCur[i]=0;
        }
        // std::cout<<std::endl;

        updateTransformCur();

        // 增量dx已经很小了
        // 则返回false，表示迭代完成
        float deltaR = sqrt(
                            pow(rad2deg(Delta_x(0, 0)), 2) +
                            pow(rad2deg(Delta_x(1, 0)), 2));
        float deltaT = sqrt(
                            pow(Delta_x(2, 0) * 100, 2));

        if (deltaR < 0.1 && deltaT < 0.1) {
            return false;
        }

        return true; 
    }
    #endif

    bool calculateTransformationCorner(int iterCount){
        int pointSelNum = laserCloudOri->points.size();

        Eigen::Vector3f v_pointOri_bk1;

        Eigen::Matrix<float, 1, 3> j_n;
        Eigen::Matrix3f H = Eigen::Matrix3f::Zero();
        Eigen::Vector3f b = Eigen::Vector3f::Zero();
        float cost = 0.0;

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
            neg_Rp_sym = -q_transformCur.toRotationMatrix()*neg_Rp_sym;
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
            cost += f_n;

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

        // Delta_x.normalize();

        // Delta_x = Delta_x;

        transformCur[2] += Delta_x[0];
        transformCur[3] += Delta_x[1];
        transformCur[4] += Delta_x[2];

        // std::cout<<iterCount<<"] "<<"transformCur : "<<transformCur[2]<<", "<<transformCur[3]<<", "<<transformCur[4]<<", "<<std::endl;
        // std::cout<<iterCount<<"] "<<"Delta_x : "<<Delta_x.transpose()<<std::endl;

        for(int i=0; i<6; i++){
            if(isnan(transformCur[i]))
                transformCur[i]=0;
        }

        updateTransformCur();

        float deltaR = sqrt(pow(rad2deg(Delta_x(0, 0)), 2));
        float deltaT = sqrt( pow(Delta_x(1, 0) * 100, 2) +
                                pow(Delta_x(2, 0) * 100, 2));

        if (deltaR < 0.1 && deltaT < 0.1) {
            return false;
        }

        lastCost = cost;

        return true; 
    }

    void calculateTransformationGround()
    {
        Eigen::Vector3f NormalCurr = PlaneEqu.block<3, 1>(0, 0);
        float dLast = PlaneEqu(3, 0);
        float pitchCurr = atan2(NormalCurr.x(), NormalCurr.z());
        float rollCurr = atan2(NormalCurr.y(), NormalCurr.z());

        Eigen::Vector3f NormalLast = PlaneEquLast.block<3, 1>(0, 0);
        float dCurr = PlaneEquLast(3, 0);
        float pitchLast = atan2(NormalLast.x(), NormalLast.z());
        float rollLast = atan2(NormalLast.y(), NormalLast.z());

        // 更新
        transformCur[0] = (rollCurr - rollLast);
        transformCur[1] = (pitchCurr - pitchLast);
        transformCur[5] = (dCurr-dLast);
        updateTransformCur();
        // std::cout<<"roll = "<<transformCur[0]/PI*180<<", "<<"pitch = "<<transformCur[1]/PI*180<<std::endl;

        PlaneEquLast = PlaneEqu;
    }
    
    void updateTransformation(){
        // std::cout<<laserCloudSurfLastNum<<std::endl;
        if (laserCloudCornerLastNum < 10 || laserCloudSurfLastNum < 100)
            return;

        #ifdef PLANE_EQU
        for (int iterCount1 = 0; iterCount1 < 25; iterCount1++) {
            laserCloudOri->clear();
            coeffSel->clear();

            // 找到对应的特征平面
            // 然后计算协方差矩阵，保存在coeffSel队列中
            // laserCloudOri中保存的是对应于coeffSel的未转换到开始时刻的原始点云数据
            findCorrespondingSurfFeatures(iterCount1);

            if (laserCloudOri->points.size() < 10)
                continue;
            // 通过面特征的匹配，计算变换矩阵
            if (calculateTransformationSurf(iterCount1) == false)
                break;
        }
        #else
        // 既然我们在预处理那花了这么多心思提取地平面，那我们索性直接使用地平面的法向量变化计算roll, pitch, 和z
        calculateTransformationGround();
        #endif

        // std::cout<<"transform : "<< transformCur[0]<<", "<<transformCur[2]<<", "<<transformCur[4]<<std::endl;
        // std::cout<<"surf coeffSel : "<<coeffSel->size();
        pointSearchCornerInd.clear();
        lastCost = 0.0;
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
        // for(int j=0; j<coeffSel->size(); j++)
        //         std::cout<<coeffSel->points[j].intensity<<", ";
        //     std::cout<<std::endl<<"--------------------------------------"<<std::endl;

        // std::cout<<"v_transformCur : "<<v_transformCur.transpose()<<std::endl;
    }

    void publishCloudsLast(){
        // 粗糙点
        // int cornerPointsLessSharpNum = cornerPointsLessSharp->points.size();
        // for (int i = 0; i < cornerPointsLessSharpNum; i++) {
        //     // TransformToEnd的作用是将k+1时刻的less特征点转移至k+1时刻的sweep的结束位置处的相机坐标系下
        //     // [输入]cornerPointsLessSharp: 投影到当前帧扫描起始坐标系(相机坐标系c系)的点
        //     // [输出]cornerPointsLessSharp: 转换到当前帧扫描结束时相机坐标系c系的点
        //     TransformToEnd(&cornerPointsLessSharp->points[i], &cornerPointsLessSharp->points[i]);
        // }

        // // 平面点
        // int surfPointsLessFlatNum = surfPointsLessFlat->points.size();
        // for (int i = 0; i < surfPointsLessFlatNum; i++) {
        //     TransformToEnd(&surfPointsLessFlat->points[i], &surfPointsLessFlat->points[i]);
        // }

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

            sensor_msgs::PointCloud2 line_point_msg;
            pcl::toROSMsg(*line_point_Cloud, line_point_msg);
            line_point_msg.header.stamp = cloudHeader.stamp;
            line_point_msg.header.frame_id = "/livox";
            pubLaserCloud_line_point.publish(line_point_msg);

            pcl::toROSMsg(*line_Cloud, line_point_msg);
            line_point_msg.header.stamp = cloudHeader.stamp;
            line_point_msg.header.frame_id = "/livox";
            pubLaserCloud_line.publish(line_point_msg);
        }
    }

    // 旋转角的累计变化量
    void integrateTransformation(){
        // 存储odom坐标系下的当前位置
        // float rx, ry, rz, tx, ty, tz;

        // AccumulateRotation作用
        // 将计算的两帧之间的位姿“累加”起来，获得相对于第一帧的旋转矩阵
        // transformSum + (-transformCur) =(rx,ry,rz)
        Eigen::AngleAxisf rollAngle(
            Eigen::AngleAxisf(transformSum[0], Eigen::Vector3f::UnitX())
        );
        Eigen::AngleAxisf pitchAngle(
            Eigen::AngleAxisf(transformSum[1], Eigen::Vector3f::UnitY())
        );
        Eigen::AngleAxisf yawAngle(
            Eigen::AngleAxisf(transformSum[2], Eigen::Vector3f::UnitZ())
        ); 
        // Eigen::AngleAxisf Angle = Eigen::AngleAxisf(
        //     sqrt(transformSum[0]*transformSum[0] + 
        //         transformSum[1]*transformSum[1] + 
        //         transformSum[2]*transformSum[2]),
        //     Eigen::Vector3f(transformSum[0], transformSum[1], transformSum[2])
        // );
        Eigen::Quaternionf q_transformSum;
        q_transformSum = rollAngle * pitchAngle * yawAngle;
        // q_transformSum = Angle;

        Eigen::Vector3f v_transformSum(transformSum[3], transformSum[4], transformSum[5]);
        // v_transformCur = Eigen::Vector3f(-transformCur[3], -transformCur[4], -transformCur[5]);
        v_transformSum = v_transformSum + q_transformSum * v_transformCur;

        q_transformSum = q_transformSum * q_transformCur;

        Eigen::Vector3f e_transformSum = q_transformSum.toRotationMatrix().eulerAngles(0, 1, 2);

        // tx = v_transformSum.x();
        // ty = v_transformSum.y();
        // tz = v_transformSum.z();

        // 计算积累的t
        transformSum[0] = e_transformSum.x();
        transformSum[1] = e_transformSum.y();
        transformSum[2] = e_transformSum.z();
        transformSum[3] = v_transformSum.x();
        transformSum[4] = v_transformSum.y();
        transformSum[5] = v_transformSum.z();
    }

    void publishOdometry(){
        // 这里的旋转轴是怎么回事？
        geometry_msgs::Quaternion geoQuat = tf::createQuaternionMsgFromRollPitchYaw(transformSum[0], transformSum[1], transformSum[2]);

        // rx,ry,rz转化为四元数发布
        laserOdometry.header.stamp = cloudHeader.stamp;
        laserOdometry.pose.pose.orientation.x = geoQuat.x;
        laserOdometry.pose.pose.orientation.y = geoQuat.y;
        laserOdometry.pose.pose.orientation.z = geoQuat.z;
        laserOdometry.pose.pose.orientation.w = geoQuat.w;
        laserOdometry.pose.pose.position.x = transformSum[3];
        laserOdometry.pose.pose.position.y = transformSum[4];
        laserOdometry.pose.pose.position.z = transformSum[5];
        pubLaserOdometry.publish(laserOdometry);

        // laserOdometryTrans 是用于tf广播
        laserOdometryTrans.stamp_ = cloudHeader.stamp;
        laserOdometryTrans.setRotation(tf::Quaternion(geoQuat.x, geoQuat.y, geoQuat.z, geoQuat.w));
        laserOdometryTrans.setOrigin(tf::Vector3(transformSum[3], transformSum[4], transformSum[5]));
        tfBroadcaster.sendTransform(laserOdometryTrans);
    }

    void runFeatureAssociation()
    {
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        // 如果有新数据进来则执行，否则不执行任何操作
        if (newSegmentedCloud && newSegmentedCloudInfo && newGroundCloud && newPlaneEqu &&
            std::abs(timeNewSegmentedCloudInfo - timeNewSegmentedCloud) < 0.05 &&
            std::abs(timeNewSegmentedCloudInfo - timeNewGroundCloud) < 0.05 &&
            std::abs(timeNewSegmentedCloudInfo - timePlaneEqu) < 0.05){

            newSegmentedCloud = false;
            newSegmentedCloudInfo = false;
            newGroundCloud = false;
            newPlaneEqu = false;
        }else{
            return;
        }

        // 主要进行的处理是将点云数据进行坐标变换，进行插补等工作
        // adjustDistortion();
        DownSizeGroudCloud();

        // 不完全按照公式进行光滑性计算，并保存结果
        calculateSmoothness();

        // 标记阻塞点??? 阻塞点是什么点???
        // 参考了csdn若愚maimai大佬的博客，这里的阻塞点指过近的点
        // 指在点云中可能出现的互相遮挡的情况
        markOccludedPoints();

        // 特征抽取，然后分别保存到cornerPointsSharp等等队列中去
        // 保存到不同的队列是不同类型的点云，进行了标记的工作，
        // 这一步中减少了点云数量，使计算量减少
        extractFeatures();

        // 发布cornerPointsSharp等4种类型的点云数据
        publishCloud();

        if (!systemInitedLM) {
            checkSystemInitialization();
            return;
        }

        // 预测位姿
        updateInitialGuess();

        // 通过特征点ICP计算更新变换
        updateTransformation();

        // 积分总变换
        integrateTransformation();

        publishOdometry();

        publishCloudsLast();   

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
        // std::cout << "solve time cost = " << time_used.count() << " seconds. " << std::endl;
        ROS_DEBUG("solve time cost = %f seconds.", time_used.count());
    }
};



int main(int argc, char** argv)
{
    ros::init(argc, argv, "lego_loam");

    ROS_INFO("\033[1;32m---->\033[0m Feature Association Started.");

    featureAssociation FA;

    ros::Rate rate(200);
    while (ros::ok())
    {
        FA.runFeatureAssociation();
        ros::spinOnce();

        rate.sleep();
    }
    
    ros::spin();
    return 0;
}
