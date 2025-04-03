//
//  Body.cpp
//  Composition
//
//  Created by SEPH LI on 07/02/2024.
//

#include "BodyData.h"
#include "cinder/app/App.h"
#include "Unproj.h"
#include "Resources.h"

using namespace ci;
using namespace std;
using namespace ml;
static const float confidenceThres = .125f;
static const float skeletonScalar  = 1.0f;
static const float skExtend        = 1.5f;

void BodyData::drawArmRaw (const Arm& arm, const float scale) {
    gl::drawSolidCircle(arm.shoulderp, 3.f * scale);
    gl::drawSolidCircle(arm.elbowp,    3.f * scale);
    gl::drawSolidCircle(arm.wristp,    3.f * scale);
    
    // connect with lines
    if (arm.shoulder_confidence > confidenceThres &&
        arm.elbow_confidence > confidenceThres &&
        arm.wrist_confidence > confidenceThres) {
        gl::drawLine(vec2(arm.shoulderp),vec2(arm.elbowp));
        gl::drawLine(vec2(arm.wristp),   vec2(arm.elbowp));
    }
}

void BodyData::drawLegRaw (const Leg& leg, const float scale) {
    gl::drawSolidCircle(leg.hipp,   3.f * scale);
    gl::drawSolidCircle(leg.kneep,  3.f * scale);
    gl::drawSolidCircle(leg.anklep, 3.f * scale);
    
    // connect with lines
    if (leg.hip_confidence > confidenceThres &&
        leg.knee_confidence > confidenceThres &&
        leg.ankle_confidence > confidenceThres) {
        gl::drawLine(vec2(leg.hipp),   vec2(leg.kneep));
        gl::drawLine(vec2(leg.anklep), vec2(leg.kneep));
    }
}

void BodyData::drawArm(const Arm& arm, const float scale){
    gl::drawSolidCircle(arm.shoulder.pos(), 3.f * scale);
    gl::drawSolidCircle(arm.elbow.pos(),    3.f * scale);
    gl::drawSolidCircle(arm.wrist.pos(),    3.f * scale);
    
    // connect with lines
    if (arm.shoulder_confidence > confidenceThres &&
        arm.elbow_confidence > confidenceThres &&
        arm.wrist_confidence > confidenceThres) {
        gl::drawLine(vec2(arm.shoulder.pos()),vec2(arm.elbow.pos()));
        gl::drawLine(vec2(arm.wrist.pos()),   vec2(arm.elbow.pos()));
    }
}

void BodyData::drawLeg(const Leg& leg, const float scale){
    gl::drawSolidCircle(leg.hip.pos(),   3.f * scale);
    gl::drawSolidCircle(leg.knee.pos(),  3.f * scale);
    gl::drawSolidCircle(leg.ankle.pos(), 3.f * scale);
    
    // connect with lines
    if (leg.hip_confidence > confidenceThres &&
        leg.knee_confidence > confidenceThres &&
        leg.ankle_confidence > confidenceThres) {
        gl::drawLine(vec2(leg.hip.pos()),   vec2(leg.knee.pos()));
        gl::drawLine(vec2(leg.ankle.pos()), vec2(leg.knee.pos()));
    }
}

void BodyData::drawArm3D(const Arm& arm) {
    gl::drawCube(arm.shoulder.pos(), vec3(.125f/6.f));
    gl::drawCube(arm.elbow.pos(),    vec3(.125f/6.f));
    gl::drawCube(arm.wrist.pos(),    vec3(.125f/6.f));
    
    // connect with lines
    if (arm.shoulder_confidence > .25f &&
        arm.elbow_confidence > .25f &&
        arm.wrist_confidence > .25f) {
        gl::drawLine(arm.shoulder.pos(), arm.elbow.pos());
        gl::drawLine(arm.wrist.pos(),    arm.elbow.pos());
    }
}

void BodyData::drawLeg3D(const Leg& leg) {
    gl::drawCube(leg.hip.pos(),   vec3(.125f/6.f));
    gl::drawCube(leg.knee.pos(),  vec3(.125f/6.f));
    gl::drawCube(leg.ankle.pos(), vec3(.125f/6.f));
    
    // connect with lines
    if (leg.hip_confidence > .25f &&
        leg.knee_confidence > .25f &&
        leg.ankle_confidence > .25f) {
        gl::drawLine(leg.hip.pos(),   leg.knee.pos());
        gl::drawLine(leg.ankle.pos(), leg.knee.pos());
    }
}

bool BodyData::validArm(const Arm& arm){
    vec2 res = arm.shoulder.pos() * arm.elbow.pos() * arm.wrist.pos();
    return res.x * res.y > 0.f;
}

bool BodyData::validLeg(const Leg& leg){
    vec2 res = leg.hip.pos() * leg.knee.pos() * leg.ankle.pos();
    return res.x * res.y > 0.f;
}

ci::vec2 BodyData::pivot(){
    return vec2( waist.pos() + neck.pos() ) / 2.f;
}

const ci::ivec2 BodyData::trans(const ci::mat3& trans, const ci::vec2& pos){
    return glm::clamp(ivec2(glm::round(trans * vec3(pos, 1.f))), ivec2(0), ivec2(255));
}

void BodyData::assignDepth( ci::Channel32fRef depthChannel, const ci::vec2& depthScale ){
    ivec2 dsize     = depthChannel->getSize();
    vec2  scalar    = vec2(dsize) / vec2(1280.f, 720.f);
    
    // waist
    {
        ivec2 p  = trans(scalar, dsize, waistp);
        waistp.z = mapDepth(*depthChannel->getData(p), depthScale);
    }
    // left arm
    {
        ivec2 p             = trans(scalar, dsize, mLeftArm.shoulderp);
        mLeftArm.shoulderp.z= glm::clamp(mapDepth(*depthChannel->getData(p), depthScale),
                                         waistp.z - skExtend, waistp.z + skExtend);
        p                   = trans(scalar, dsize, mLeftArm.elbowp);
        mLeftArm.elbowp.z   = glm::clamp(mapDepth(*depthChannel->getData(p), depthScale),
                                         waistp.z - skExtend, waistp.z + skExtend);
        p                   = trans(scalar, dsize, mLeftArm.wristp);
        mLeftArm.wristp.z   = glm::clamp(mapDepth(*depthChannel->getData(p), depthScale),
                                         waistp.z - skExtend, waistp.z + skExtend);
    }
    // right arm
    {
        ivec2 p              = trans(scalar, dsize, mRightArm.shoulderp);
        mRightArm.shoulderp.z= glm::clamp(mapDepth(*depthChannel->getData(p), depthScale),
                                          waistp.z - skExtend, waistp.z + skExtend);
        p                    = trans(scalar, dsize, mRightArm.elbowp);
        mRightArm.elbowp.z   = glm::clamp(mapDepth(*depthChannel->getData(p), depthScale),
                                          waistp.z - skExtend, waistp.z + skExtend);
        p                    = trans(scalar, dsize, mRightArm.wristp);
        mRightArm.wristp.z   = glm::clamp(mapDepth(*depthChannel->getData(p), depthScale),
                                          waistp.z - skExtend, waistp.z + skExtend);
    }
    // left arm
    {
        ivec2 p          = trans(scalar, dsize, mLeftLeg.hipp);
        mLeftLeg.hipp.z  = glm::clamp(mapDepth(*depthChannel->getData(p), depthScale),
                                      waistp.z - skExtend, waistp.z + skExtend);
        p                = trans(scalar, dsize, mLeftLeg.kneep);
        mLeftLeg.kneep.z = glm::clamp(mapDepth(*depthChannel->getData(p), depthScale),
                                      waistp.z - skExtend, waistp.z + skExtend);
        p                = trans(scalar, dsize, mLeftLeg.anklep);
        mLeftLeg.anklep.z= glm::clamp(mapDepth(*depthChannel->getData(p), depthScale),
                                      waistp.z - skExtend, waistp.z + skExtend);
    }
    // right arm
    {
        ivec2 p          = trans(scalar, dsize, mRightLeg.hipp);
        mRightLeg.hipp.z = glm::clamp(mapDepth(*depthChannel->getData(p), depthScale),
                                      waistp.z - skExtend, waistp.z + skExtend);
        p                = trans(scalar, dsize, mRightLeg.kneep);
        mRightLeg.kneep.z= glm::clamp(mapDepth(*depthChannel->getData(p), depthScale),
                                      waistp.z - skExtend, waistp.z + skExtend);
        p                = trans(scalar, dsize, mRightLeg.anklep);
        mRightLeg.anklep.z=glm::clamp(mapDepth(*depthChannel->getData(p), depthScale),
                                      waistp.z - skExtend, waistp.z + skExtend);
    }
    // head
    {
        ivec2 p = trans(scalar, dsize, nosep);
        nosep.z = glm::clamp(mapDepth(*depthChannel->getData(p), depthScale),
                             waistp.z - skExtend, waistp.z + skExtend);
        p       = trans(scalar, dsize, neckp);
        neckp.z = glm::clamp(mapDepth(*depthChannel->getData(p), depthScale),
                             waistp.z - skExtend, waistp.z + skExtend);
        
        p           = trans(scalar, dsize, left_earp);
        left_earp.z = glm::clamp(mapDepth(*depthChannel->getData(p), depthScale),
                                 waistp.z - skExtend, waistp.z + skExtend);
        p           = trans(scalar, dsize, left_eyep);
        left_eyep.z = glm::clamp(mapDepth(*depthChannel->getData(p), depthScale),
                                 waistp.z - skExtend, waistp.z + skExtend);
        p           = trans(scalar, dsize, right_earp);
        right_earp.z= glm::clamp(mapDepth(*depthChannel->getData(p), depthScale),
                                 waistp.z - skExtend, waistp.z + skExtend);
        p           = trans(scalar, dsize, right_eyep);
        right_eyep.z= glm::clamp(mapDepth(*depthChannel->getData(p), depthScale),
                                 waistp.z - skExtend, waistp.z + skExtend);
    }
}

void BodyData::resetArm (Arm& arm, const Arm& other,
                         const SjUtil::Calibrater::CameraData &cam) {
    arm.shoulderp   = other.shoulderp;
    arm.shoulderp.z = waistp.z + skeletonScalar * (arm.shoulderp.z - waistp.z);
    arm.elbowp      = other.elbowp;
    arm.elbowp.z    = waistp.z + skeletonScalar * (arm.elbowp.z - waistp.z);
    arm.wristp      = other.wristp;
    arm.wristp.z    = waistp.z + skeletonScalar * (arm.wristp.z - waistp.z);
    
    arm.shoulder.reset(SjUtil::Unproj::Unproject(arm.shoulderp, cam));
    arm.elbow.reset   (SjUtil::Unproj::Unproject(arm.elbowp,    cam));
    arm.wrist.reset   (SjUtil::Unproj::Unproject(arm.wristp,    cam));
    
    arm.arm_confidence      = other.arm_confidence;
    arm.elbow_confidence    = other.elbow_confidence;
    arm.shoulder_confidence = other.shoulder_confidence;
    arm.wrist_confidence    = other.wrist_confidence;
}

void BodyData::resetLeg (Leg& leg, const Leg& other,
                         const SjUtil::Calibrater::CameraData &cam) {
    leg.hipp    = other.hipp;
    leg.hipp.z  = waistp.z + skeletonScalar * (leg.hipp.z - waistp.z);
    leg.kneep   = other.kneep;
    leg.kneep.z = waistp.z + skeletonScalar * (leg.kneep.z - waistp.z);
    leg.anklep  = other.anklep;
    leg.anklep.z= waistp.z + skeletonScalar * (leg.anklep.z - waistp.z);
    
    leg.hip.reset   (SjUtil::Unproj::Unproject(leg.hipp,   cam));
    leg.knee.reset  (SjUtil::Unproj::Unproject(leg.kneep,  cam));
    leg.ankle.reset (SjUtil::Unproj::Unproject(leg.anklep, cam));
        
    leg.ankle_confidence = other.ankle_confidence;
    leg.hip_confidence   = other.hip_confidence;
    leg.knee_confidence  = other.knee_confidence;
    leg.leg_confidence   = other.leg_confidence;
}

void BodyData::applyArm (Arm& arm, const Arm& other) {
    arm.shoulderp   = other.shoulderp;
    arm.shoulderp.z = waistp.z + skeletonScalar * (arm.shoulderp.z - waistp.z);
    arm.elbowp      = other.elbowp;
    arm.elbowp.z    = waistp.z + skeletonScalar * (arm.elbowp.z - waistp.z);
    arm.wristp      = other.wristp;
    arm.wristp.z    = waistp.z + skeletonScalar * (arm.wristp.z - waistp.z);
    
    arm.arm_confidence      = other.arm_confidence;
    arm.elbow_confidence    = other.elbow_confidence;
    arm.shoulder_confidence = other.shoulder_confidence;
    arm.wrist_confidence    = other.wrist_confidence;
}

void BodyData::applyLeg (Leg& leg, const Leg& other) {
    leg.hipp    = other.hipp;
    leg.hipp.z  = waistp.z + skeletonScalar * (leg.hipp.z - waistp.z);
    leg.kneep   = other.kneep;
    leg.kneep.z = waistp.z + skeletonScalar * (leg.kneep.z - waistp.z);
    leg.anklep  = other.anklep;
    leg.anklep.z= waistp.z + skeletonScalar * (leg.anklep.z - waistp.z);
    
    leg.ankle_confidence = other.ankle_confidence;
    leg.hip_confidence   = other.hip_confidence;
    leg.knee_confidence  = other.knee_confidence;
    leg.leg_confidence   = other.leg_confidence;
}

void BodyData::updateArm(Arm& arm, const float dt,
                         const SjUtil::Calibrater::CameraData &cam){
    if(arm.shoulder_confidence >= confidenceThres)
        arm.shoulder.update(SjUtil::Unproj::Unproject(arm.shoulderp, cam), dt);
    if(arm.elbow_confidence >= confidenceThres)
        arm.elbow.update   (SjUtil::Unproj::Unproject(arm.elbowp, cam), dt);
    if(arm.wrist_confidence >= confidenceThres)
        arm.wrist.update   (SjUtil::Unproj::Unproject(arm.wristp, cam), dt);
}

void BodyData::updateLeg(Leg& leg, const float dt,
                         const SjUtil::Calibrater::CameraData &cam){
    if(leg.hip_confidence >= confidenceThres)
        leg.hip.update   (SjUtil::Unproj::Unproject(leg.hipp, cam), dt);
    if(leg.knee_confidence >= confidenceThres)
        leg.knee.update  (SjUtil::Unproj::Unproject(leg.kneep, cam), dt);
    if(leg.ankle_confidence >= confidenceThres)
        leg.ankle.update (SjUtil::Unproj::Unproject(leg.anklep, cam), dt);
    else leg.ankle.update(SjUtil::Unproj::Unproject(2.f * leg.kneep - leg.hipp, cam), dt);
}

void BodyData::reset( BodyData& other,
                      const SjUtil::Calibrater::CameraData &cam ) {
    waistp      = other.waistp;
    neckp       = other.neckp;
    nosep       = other.nosep;
    left_earp   = other.left_earp;
    left_eyep   = other.left_eyep;
    right_earp  = other.right_earp;
    right_eyep  = other.right_eyep;
    
    waist.reset     (SjUtil::Unproj::Unproject(waistp, cam));
    neck.reset      (SjUtil::Unproj::Unproject(neckp, cam));
    nose.reset      (SjUtil::Unproj::Unproject(nosep, cam));
    left_ear.reset  (SjUtil::Unproj::Unproject(left_earp, cam));
    left_eye.reset  (SjUtil::Unproj::Unproject(left_eyep, cam));
    right_ear.reset (SjUtil::Unproj::Unproject(right_earp, cam));
    right_eye.reset (SjUtil::Unproj::Unproject(right_eyep, cam));
    
    resetArm(mLeftArm,  other.mLeftArm,  cam);
    resetArm(mRightArm, other.mRightArm, cam);
    resetLeg(mLeftLeg,  other.mLeftLeg,  cam);
    resetLeg(mRightLeg, other.mRightLeg, cam);
    
    waist_confidence    = other.waist_confidence;
    neck_confidence     = other.neck_confidence;
    nose_confidence     = other.nose_confidence;
    left_ear_confidence = other.left_ear_confidence;
    left_eye_confidence = other.left_eye_confidence;
    right_ear_confidence= other.right_ear_confidence;
    right_eye_confidence= other.right_eye_confidence;
}

void BodyData::update( BodyData& other ) {
    waistp      = other.waistp;
    neckp       = other.neckp;
    nosep       = other.nosep;
    left_earp   = other.left_earp;
    left_eyep   = other.left_eyep;
    right_earp  = other.right_earp;
    right_eyep  = other.right_eyep;
    
    applyArm(mLeftArm, other.mLeftArm);
    applyArm(mRightArm,other.mRightArm);
    applyLeg(mLeftLeg, other.mLeftLeg);
    applyLeg(mRightLeg,other.mRightLeg);

    waist_confidence    = other.waist_confidence;
    neck_confidence     = other.neck_confidence;
    nose_confidence     = other.nose_confidence;
    left_ear_confidence = other.left_ear_confidence;
    left_eye_confidence = other.left_eye_confidence;
    right_ear_confidence= other.right_ear_confidence;
    right_eye_confidence= other.right_eye_confidence;
}

void BodyData::mapWorld(const float dt, const SjUtil::Calibrater::CameraData &cam) {
    if(waist_confidence >= confidenceThres)
        waist.update    (SjUtil::Unproj::Unproject(waistp, cam), dt);
    
    if(neck_confidence >= confidenceThres)
        neck.update     (SjUtil::Unproj::Unproject(neckp, cam), dt);
    else neck.update(SjUtil::Unproj::Unproject(waistp + .85f * (waistp - .5f * (mLeftLeg.hipp + mRightLeg.hipp)), cam), dt);
    
    if(nose_confidence >= confidenceThres)
        nose.update     (SjUtil::Unproj::Unproject(nosep, cam), dt);
    else nose.update(SjUtil::Unproj::Unproject(waistp + .5f * (nosep - waistp), cam), dt);
    
    if(left_ear_confidence >= confidenceThres)
        left_ear.update (SjUtil::Unproj::Unproject(left_earp, cam), dt);
    if(left_eye_confidence >= confidenceThres)
        left_eye.update (SjUtil::Unproj::Unproject(left_eyep, cam), dt);
    if(right_ear_confidence >= confidenceThres)
        right_ear.update(SjUtil::Unproj::Unproject(right_earp, cam), dt);
    if(right_eye_confidence >= confidenceThres)
        right_eye.update(SjUtil::Unproj::Unproject(right_eyep, cam), dt);
    
    updateArm(mLeftArm, dt, cam);
    updateArm(mRightArm,dt, cam);
    updateLeg(mLeftLeg, dt, cam);
    updateLeg(mRightLeg,dt, cam);
}

void BodyData::setArmNoise(Arm& arm, const float q, const float r) {
    arm.elbow.setMeasurementNoise(q);
    arm.elbow.setProcessNoise(r);
    arm.shoulder.setMeasurementNoise(q);
    arm.shoulder.setProcessNoise(r);
    arm.wrist.setMeasurementNoise(q);
    arm.wrist.setProcessNoise(r);
}

void BodyData::setLegNoise(Leg& leg, const float q, const float r) {
    leg.ankle.setMeasurementNoise(q);
    leg.ankle.setProcessNoise(r);
    leg.hip.setMeasurementNoise(q);
    leg.hip.setProcessNoise(r);
    leg.knee.setMeasurementNoise(q);
    leg.knee.setProcessNoise(r);
}

void BodyData::setNoise(const float q, const float r){
    setArmNoise(mLeftArm,  q, r);
    setArmNoise(mRightArm, q, r);
    setLegNoise(mLeftLeg,  q, r);
    setLegNoise(mRightLeg, q, r);
    
    waist.setMeasurementNoise(q);
    waist.setProcessNoise    (r);
    neck.setMeasurementNoise (q);
    neck.setProcessNoise     (r);
    nose.setMeasurementNoise (q);
    nose.setProcessNoise     (r);
    
    left_ear.setMeasurementNoise (q);
    left_ear.setProcessNoise     (r);
    left_eye.setMeasurementNoise (q);
    left_eye.setProcessNoise     (r);
    right_eye.setMeasurementNoise (q);
    right_eye.setProcessNoise     (r);
    right_ear.setMeasurementNoise (q);
    right_ear.setProcessNoise     (r);
}

float BodyData::confidence() const {
    return mLeftArm.elbow_confidence + mLeftArm.shoulder_confidence + mLeftArm.wrist_confidence +
           mRightArm.elbow_confidence + mRightArm.shoulder_confidence + mRightArm.wrist_confidence +
           mLeftLeg.ankle_confidence + mLeftLeg.hip_confidence + mLeftLeg.knee_confidence +
           mRightLeg.ankle_confidence + mRightLeg.hip_confidence + mRightLeg.knee_confidence +
           waist_confidence + neck_confidence;
}

bool BodyData::isvalid(const float cutoff) {
#if PROCESS_DEBUG
    return true;
#endif
    
    return  glm::min(
                     glm::max(glm::max(mLeftArm.wrist_confidence,
                              mLeftArm.elbow_confidence),
                              mLeftArm.shoulder_confidence),
                    glm::max(glm::max(mRightArm.wrist_confidence,
                              mRightArm.elbow_confidence),
                             mRightArm.shoulder_confidence)
                     ) > confidenceThres &&
            glm::max(glm::min(mLeftLeg.hip_confidence, mLeftLeg.knee_confidence),
                     mLeftLeg.ankle_confidence) > confidenceThres &&
            glm::max(glm::min(mLeftLeg.hip_confidence, mLeftLeg.knee_confidence),
                     mLeftLeg.ankle_confidence) > confidenceThres &&
    waist_confidence > confidenceThres;// && neck_confidence > confidenceThres;
}

void BodyData::draw3D( const ci::vec3& scalar, const float offset ){
    vec3 scale = vec3(3.f * scalar.x, -3.f * scalar.x, scalar.z);
    vec3 lift  = vec3(0.f, offset, scalar.y);
    
    gl::ScopedModelMatrix scpModel(glm::translate(lift) * glm::scale(scale));
    
    {
        drawArm3D(mLeftArm);
        drawArm3D(mRightArm);
    }
    
    {
        drawLeg3D(mLeftLeg);
        drawLeg3D(mRightLeg);
    }
    
    {
        gl::drawCube(waist.pos(), vec3(.125f/6.f));
        gl::drawLine(mRightLeg.hip.pos(), waist.pos());
        gl::drawLine(mLeftLeg.hip.pos(),  waist.pos());
    }
    
    {
        gl::drawCube(neck.pos(),      vec3(.125f/6.f));
        gl::drawCube(nose.pos(),      vec3(.125f/6.f));
        gl::drawCube(left_ear.pos(),  vec3(.075f/6.f));
        gl::drawCube(left_eye.pos(),  vec3(.075f/6.f));
        gl::drawCube(right_ear.pos(), vec3(.075f/6.f));
        gl::drawCube(right_eye.pos(), vec3(.075f/6.f));
        
        ///connect with lines
        if (neck_confidence > .15f &&
            nose_confidence > .15f &&
            right_eye_confidence > .15f &&
            left_eye_confidence > .15f) {
            gl::drawLine(neck.pos(), nose.pos());
            gl::drawLine(right_eye.pos(), left_eye.pos());
        }
        
        gl::drawLine(waist.pos(), neck.pos());
        gl::drawLine(mRightArm.shoulder.pos(), neck.pos());
        gl::drawLine(neck.pos(), mLeftArm.shoulder.pos());
    }
}

void BodyData::drawNoColor(const float aspectRatio) {
    vec2 scale = vec2(app::toPixels(app::getWindowSize())) / vec2(aspectRatio, 1.f);
    gl::ScopedModelMatrix scpMat(glm::scale(vec3(scale, 1.f)) * glm::translate(vec3(.5f*aspectRatio,.0f,.0f)));
    
    {
        drawArmRaw(mLeftArm, 1.f/scale.x);
        drawArmRaw(mRightArm,1.f/scale.x);
    }
    
    {
        drawLegRaw(mLeftLeg, 1.f/scale.x);
        drawLegRaw(mRightLeg,1.f/scale.x);
    }
    
    {
        gl::drawSolidCircle(waist.pos(), 3.f / scale.x);
        gl::drawLine(vec2(mRightLeg.hip.pos()), vec2(waist.pos()));
        gl::drawLine(vec2(mLeftLeg.hip.pos()),  vec2(waist.pos()));
    }
    
    {
        gl::drawSolidCircle(neck.pos(),      3.f / scale.x);
        gl::drawSolidCircle(nose.pos(),      3.f / scale.x);
        gl::drawSolidCircle(left_ear.pos(),  1.f / scale.x);
        gl::drawSolidCircle(left_eye.pos(),  1.f / scale.x);
        gl::drawSolidCircle(right_ear.pos(), 1.f / scale.x);
        gl::drawSolidCircle(right_eye.pos(), 1.f / scale.x);
        
        ///connect with lines
        if (neck_confidence > .15f &&
            nose_confidence > .15f &&
            right_eye_confidence > .15f &&
            left_eye_confidence > .15f) {
            gl::drawLine(vec2(neck.pos()),      vec2(nose.pos()));
            gl::drawLine(vec2(right_eye.pos()), vec2(left_eye.pos()));
        }
        
        gl::drawLine(vec2(waist.pos()),               vec2(neck.pos()));
        gl::drawLine(vec2(mRightArm.shoulder.pos()),  vec2(neck.pos()));
        gl::drawLine(vec2(neck.pos()),
                     vec2(mLeftArm.shoulder.pos()));
    }
}

void BodyData::draw(const float aspectRatio, const float cutoff) {
    if(!isvalid(cutoff)) return;
    vec2 scale = vec2(app::toPixels(app::getWindowSize())) / vec2(aspectRatio, 1.f);
    gl::ScopedModelMatrix scpMat(glm::scale(vec3(scale, 1.f)) * glm::translate(vec3(.5f*aspectRatio,.0f,.0f)));
    
    {
        gl::ScopedColor scpColor(Color::hex(0xff0000));
        drawArm(mLeftArm, 1.f/scale.x);
        drawArm(mRightArm,1.f/scale.x);
    }
    
    {
        gl::ScopedColor scpColor(Color::hex(0x00ff00));
        drawLeg(mLeftLeg, 1.f/scale.x);
        drawLeg(mRightLeg,1.f/scale.x);
    }
    
    {
        gl::ScopedColor scpColor(Color::hex(0x0000ff));
        gl::drawSolidCircle(waistp, 3.f / scale.x);
        gl::drawLine(vec2(mRightLeg.hipp), vec2(waist.pos()));
        gl::drawLine(vec2(mLeftLeg.hipp),  vec2(waist.pos()));
    }
    
    {
        gl::ScopedColor scpColor(Color::hex(0xffff00));
        
        gl::drawSolidCircle(neckp,      3.f / scale.x);
        gl::drawSolidCircle(nosep,      3.f / scale.x);
        gl::drawSolidCircle(left_earp,  1.f / scale.x);
        gl::drawSolidCircle(left_eyep,  1.f / scale.x);
        gl::drawSolidCircle(right_earp, 1.f / scale.x);
        gl::drawSolidCircle(right_eyep, 1.f / scale.x);
        
        ///connect with lines
        if (neck_confidence > .15f &&
            nose_confidence > .15f &&
            right_eye_confidence > .15f &&
            left_eye_confidence > .15f) {
            gl::drawLine(vec2(neckp),      vec2(nose.pos()));
            gl::drawLine(vec2(right_eyep), vec2(left_eye.pos()));
        }
        
        gl::drawLine(vec2(waistp),               vec2(neck.pos()));
        gl::drawLine(vec2(mRightArm.shoulderp),  vec2(neck.pos()));
        gl::drawLine(vec2(neckp), vec2(mLeftArm.shoulderp));
    }
}

