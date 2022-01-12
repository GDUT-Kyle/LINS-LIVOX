#ifndef _INCLUDE_TRANSFORM_H
#define _INCLUDE_TRANSFORM_H
#include <eigen3/Eigen/Dense>

class Transform
{
public:
    Eigen::Quaternionf q_rotation;
    Eigen::Vector3f v_transition;
public:
    Transform(){}
    Transform(Eigen::Quaternionf q_, Eigen::Vector3f v_):q_rotation(q_), v_transition(v_){}
    Transform(Eigen::Matrix<float, 4, 4> mat_){
        q_rotation = mat_.block<3, 3>(0, 0);
        v_transition = mat_.block<3, 1>(3, 0);
    }
    Transform(Eigen::Matrix3f r_, Eigen::Vector3f v_):v_transition(v_){
        q_rotation = r_;
    }
    Transform(Eigen::AngleAxisf axis_, Eigen::Vector3f v_):v_transition(v_){
        q_rotation = axis_;
    }
    ~Transform(){};

    Eigen::Quaternionf getQuaternionf()
    {
        return q_rotation;
    }

    Eigen::Vector3f getTransition()
    {
        return v_transition;
    }

    Eigen::Matrix3f getRotationMatrix()
    {
        return q_rotation.toRotationMatrix();
    }

    Eigen::Vector3f getEulurAngleRPY()
    {
        return q_rotation.toRotationMatrix().eulerAngles(0, 1, 2);
    }

    Eigen::Matrix<float, 4, 4> getTransformMatrix()
    {
        Eigen::Matrix<float, 4, 4> Mat;
        Mat.setIdentity();
        Mat.block<3, 3>(0, 0) = q_rotation.toRotationMatrix();
        Mat.block<3, 1>(3, 0) = v_transition;
        return Mat;
    }

    Transform rightMul(Transform& transform_) const
    {
        Transform newTransform(q_rotation*(transform_.getQuaternionf()), 
                        v_transition+q_rotation*(transform_.getTransition()));
        return newTransform;
    }
};

#endif