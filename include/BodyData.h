//
//  BodyData.h
//  Composition
//
//  Created by SEPH LI on 07/02/2022.
//

#ifndef BodyData_h
#define BodyData_h
#include "cinder/gl/gl.h"
#include "KalmanFilter.h"
#include "Calibrater.h"

//! Helper class stores skeleton data, with debugging drawing functions
namespace ml {
class BodyData {
public:
    struct Arm {
    public:
        Kalman3f shoulder;
        Kalman3f elbow;
        Kalman3f wrist;
        
        ci::vec3 shoulderp, elbowp, wristp;
        
        float   shoulder_confidence;
        float   elbow_confidence;
        float   wrist_confidence;
        float   arm_confidence;
    };
    
    struct Leg {
    public:
        Kalman3f hip;
        Kalman3f knee;
        Kalman3f ankle;
        
        ci::vec3 hipp, kneep, anklep;
        
        float hip_confidence;
        float knee_confidence;
        float ankle_confidence;
        float leg_confidence;
    };
    
protected:
    inline float mapDepth(float depth, const ci::vec2& range){
        return range.x + range.y * (1.f - depth);
    }
    
    void drawArm (const Arm& arm, const float scale);
    void drawLeg (const Leg& leg, const float scale);
    void drawArmRaw (const Arm& arm, const float scale);
    void drawLegRaw (const Leg& leg, const float scale);
    
    bool validArm(const Arm& arm);
    bool validLeg(const Leg& leg);
    
    void drawArm3D(const Arm& arm);
    void drawLeg3D(const Leg& leg);
    
    void resetArm (Arm& arm, const Arm& other,
                   const SjUtil::Calibrater::CameraData &cam);
    void resetLeg (Leg& leg, const Leg& other,
                   const SjUtil::Calibrater::CameraData &cam);
    void applyArm (Arm& arm, const Arm& other);
    void applyLeg (Leg& leg, const Leg& other);
    
    void updateArm(Arm& arm, const float dt,
                   const SjUtil::Calibrater::CameraData &cam);
    void updateLeg(Leg& leg, const float dt,
                   const SjUtil::Calibrater::CameraData &cam);
    
    void setArmNoise(Arm& arm, const float q, const float r);
    void setLegNoise(Leg& leg, const float q, const float r);
    const ci::ivec2 trans(const ci::mat3& trans, const ci::vec2& pos);
    inline const ci::ivec2 trans(
        const ci::vec2 scale, const ci::ivec2 limit, const ci::vec2 pos){
        ci::ivec2 p = pos * scale;
        return glm::clamp(p, ci::ivec2(0), limit);
    }
    
public:
    uint32_t camIdx;
    // ARM
    Arm mLeftArm, mRightArm;
    
    // LEG
    Leg mLeftLeg, mRightLeg;
    
    //WAIST
    Kalman3f waist;
    ci::vec3 waistp;
    float waist_confidence;
    float torso_confidence;
    
    //HEAD
    Kalman3f neck;
    Kalman3f nose;
    Kalman3f left_ear;
    Kalman3f left_eye;
    Kalman3f right_ear;
    Kalman3f right_eye;
    
    ci::vec3 neckp, nosep,
             left_earp, left_eyep,
             right_earp, right_eyep;
    
    float neck_confidence;
    float nose_confidence;
    float left_ear_confidence;
    float left_eye_confidence;
    float right_ear_confidence;
    float right_eye_confidence;
    float head_confidence;
    
    // assign depth with normalised values
    void assignDepth( ci::Channel32fRef depthChannel,
                      const ci::vec2& depthScale );
    float confidence() const;
    void reset   (BodyData& other,
                  const SjUtil::Calibrater::CameraData &cam);
    void update  (BodyData& other);
    void setNoise(const float q, const float r);
    
    void mapWorld(const float dt,
                  const SjUtil::Calibrater::CameraData &cam);
    
    void draw       ( const float aspectRatio, const float cutoff );
    void drawNoColor( const float aspectRatio );
    void draw3D     ( const ci::vec3& scalar, const float offset );
    bool isvalid    ( const float cutoff );
    ci::vec2 pivot  ();
};
};
#endif /* BodyData_h */
