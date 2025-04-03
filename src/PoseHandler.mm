//
//  Body.cpp
//  Composition
//
//  Created by SEPH LI on 07/02/2024.
//

#include "PoseHandler.h"

#include "cinder/Cinder.h"
#include "cinder/gl/gl.h"
#include "cinder/app/App.h"
#include "cinder/Log.h"

#import  <CoreML/CoreML.h>
#import  <Vision/Vision.h>

#define NORMALIZE_RES 0

using namespace ci;
using namespace std;
using namespace ml;

API_AVAILABLE(macos(11.0)) VNDetectHumanBodyPoseRequest  *_mlRequestBody;
NSDictionary        *_rqDictionary;
ci::ivec2           _inputSize;
vector<BodyData>    _data;
NSArray             *_rqBodyArray;

CGImageRef PoseTracker::convetSurface( ci::Surface8uRef surf ){
    auto src                        = *surf;
    _inputSize                      = surf->getSize();
    uint width                      = surf->getWidth();
    uint height                     = surf->getHeight();
    unsigned char *pixels           = (unsigned char*)malloc(height*width*4);
    CGColorSpaceRef colorSpaceRef   = CGColorSpaceCreateDeviceRGB();
    CGContextRef context            = CGBitmapContextCreate
        (pixels, width, height, 8, 4*width,
         colorSpaceRef, kCGImageAlphaPremultipliedLast);
    CGColorSpaceRelease(colorSpaceRef);
    
    auto pixelIter = src.getIter();
    int x = 0, y = 0;
    while( pixelIter.line() ) {
        x = 0;
        while( pixelIter.pixel() ) {
            int idx         = (width*y+x)*4;
            pixels[idx+0]   = pixelIter.r();
            pixels[idx+1]   = pixelIter.g();
            pixels[idx+2]   = pixelIter.b();
            pixels[idx+3]   = pixelIter.a();
            x ++;
        }
        y ++;
    }
    
    auto imageRef = CGBitmapContextCreateImage(context);
    CGContextRelease(context);
    free(pixels);
    
    return imageRef;
}

void PoseTracker::init(){
    if (@available(macOS 11.0, *)) {
        _mlRequestBody = [[VNDetectHumanBodyPoseRequest alloc]
            initWithCompletionHandler: ^(VNRequest *request, NSError *error){
            
            if(error != nil){
                NSString *err = error.localizedDescription;
                CI_LOG_E( "BodyTracking returned with error:" << [err UTF8String] );
                return;
            }
                          
            vector<BodyData> tmpData;
            auto res = request.results; //[request.results copy];
            if(res.count > 0){
                NSError *err;
                for(VNHumanBodyPoseObservation *observation in res){
                    // get all points
                    NSDictionary <VNHumanBodyPoseObservationJointName, VNRecognizedPoint *> *allPts = [observation recognizedPointsForJointsGroupName:VNHumanBodyPoseObservationJointsGroupNameAll error:&err];
                        
                    if(error != nil){
                        NSString *err = error.localizedDescription;
                        CI_LOG_E( "Body Data error: " << [err UTF8String]);
                        continue;
                    }

                    BodyData data;
#if NORMALIZE_RES
                    vec2 aspect = vec2( (float)_inputSize.x / _inputSize.y, 1.f );
                    aspect.y    = aspect.x/2.f;
#else
                    const vec2 size = vec2(1280.f, 720.f);
#endif
                    {
                        // left arm
                        VNRecognizedPoint *shoulder = [allPts objectForKey:VNHumanBodyPoseObservationJointNameLeftShoulder];
                        VNRecognizedPoint *elbow = [allPts objectForKey:VNHumanBodyPoseObservationJointNameLeftElbow];
                        VNRecognizedPoint *wrist = [allPts objectForKey:VNHumanBodyPoseObservationJointNameLeftWrist];
   
#if NORMALIZE_RES
                        data.mLeftArm.shoulderp.x = shoulder.location.x * aspect.x - aspect.y;
                        data.mLeftArm.shoulderp.y = (1.f - shoulder.location.y);
                        data.mLeftArm.elbowp.x    = elbow.location.x * aspect.x - aspect.y;
                        data.mLeftArm.elbowp.y    = (1.f - elbow.location.y);
                        data.mLeftArm.wristp.x    = wrist.location.x * aspect.x - aspect.y;
                        data.mLeftArm.wristp.y    = (1.f - wrist.location.y);
#else
                        data.mLeftArm.shoulderp.x = shoulder.location.x * size.x;
                        data.mLeftArm.shoulderp.y = (1.f - shoulder.location.y) * size.y;
                        data.mLeftArm.elbowp.x    = elbow.location.x * size.x;
                        data.mLeftArm.elbowp.y    = (1.f - elbow.location.y) * size.y;
                        data.mLeftArm.wristp.x    = wrist.location.x * size.x;
                        data.mLeftArm.wristp.y    = (1.f - wrist.location.y) * size.y;
#endif
                        data.mLeftArm.shoulder_confidence = shoulder.confidence;
                        data.mLeftArm.elbow_confidence    = elbow.confidence;
                        data.mLeftArm.wrist_confidence    = wrist.confidence;
                    }
                                
                    // right arm
                    {
                        VNRecognizedPoint *shoulder = [allPts objectForKey:VNHumanBodyPoseObservationJointNameRightShoulder];
                        VNRecognizedPoint *elbow = [allPts objectForKey:VNHumanBodyPoseObservationJointNameRightElbow];
                        VNRecognizedPoint *wrist = [allPts objectForKey:VNHumanBodyPoseObservationJointNameRightWrist];
                        
#if NORMALIZE_RES
                        data.mRightArm.shoulderp.x = shoulder.location.x * aspect.x - aspect.y;
                        data.mRightArm.shoulderp.y = (1.f - shoulder.location.y);
                        data.mRightArm.elbowp.x    = elbow.location.x * aspect.x - aspect.y;
                        data.mRightArm.elbowp.y    = (1.f - elbow.location.y);
                        data.mRightArm.wristp.x    = wrist.location.x * aspect.x - aspect.y;
                        data.mRightArm.wristp.y    = (1.f - wrist.location.y);
#else
                        data.mRightArm.shoulderp.x = shoulder.location.x * size.x;
                        data.mRightArm.shoulderp.y = (1.f - shoulder.location.y) * size.y;
                        data.mRightArm.elbowp.x    = elbow.location.x * size.x;
                        data.mRightArm.elbowp.y    = (1.f - elbow.location.y) * size.y;
                        data.mRightArm.wristp.x    = wrist.location.x * size.x;
                        data.mRightArm.wristp.y    = (1.f - wrist.location.y) * size.y;
#endif
                        data.mRightArm.shoulder_confidence = shoulder.confidence;
                        data.mRightArm.elbow_confidence    = elbow.confidence;
                        data.mRightArm.wrist_confidence    = wrist.confidence;
                    }

                    // left leg
                    {
                        VNRecognizedPoint *hip = [allPts objectForKey:VNHumanBodyPoseObservationJointNameLeftHip];
                        VNRecognizedPoint *knee = [allPts objectForKey:VNHumanBodyPoseObservationJointNameLeftKnee];
                        VNRecognizedPoint *ankle = [allPts objectForKey:VNHumanBodyPoseObservationJointNameLeftAnkle];
                        
#if NORMALIZE_RES
                        data.mLeftLeg.hipp.x   = hip.location.x * aspect.x - aspect.y;
                        data.mLeftLeg.hipp.y   = (1.f - hip.location.y);
                        data.mLeftLeg.kneep.x  = knee.location.x * aspect.x - aspect.y;
                        data.mLeftLeg.kneep.y  = (1.f - knee.location.y);
                        data.mLeftLeg.anklep.x = ankle.location.x * aspect.x - aspect.y;
                        data.mLeftLeg.anklep.y = (1.f - ankle.location.y);
#else
                        data.mLeftLeg.hipp.x   = hip.location.x * size.x;
                        data.mLeftLeg.hipp.y   = (1.f - hip.location.y) * size.y;
                        data.mLeftLeg.kneep.x  = knee.location.x * size.x;
                        data.mLeftLeg.kneep.y  = (1.f - knee.location.y) * size.y;
                        data.mLeftLeg.anklep.x = ankle.location.x * size.x;
                        data.mLeftLeg.anklep.y = (1.f - ankle.location.y) * size.y;
#endif
                        data.mLeftLeg.hip_confidence = hip.confidence;
                        data.mLeftLeg.knee_confidence = knee.confidence;
                        data.mLeftLeg.ankle_confidence = ankle.confidence;
                    }
                                
                    // right leg
                    {
                        VNRecognizedPoint *hip = [allPts objectForKey:VNHumanBodyPoseObservationJointNameRightHip];
                        VNRecognizedPoint *knee = [allPts objectForKey:VNHumanBodyPoseObservationJointNameRightKnee];
                        VNRecognizedPoint *ankle = [allPts objectForKey:VNHumanBodyPoseObservationJointNameRightAnkle];
#if NORMALIZE_RES
                        data.mRightLeg.hipp.x   = hip.location.x * aspect.x - aspect.y;
                        data.mRightLeg.hipp.y   = (1.f - hip.location.y);
                        data.mRightLeg.kneep.x  = knee.location.x * aspect.x - aspect.y;
                        data.mRightLeg.kneep.y  = (1.f - knee.location.y);
                        data.mRightLeg.anklep.x = ankle.location.x * aspect.x - aspect.y;
                        data.mRightLeg.anklep.y = (1.f - ankle.location.y);
#else
                        data.mRightLeg.hipp.x   = hip.location.x * size.x;
                        data.mRightLeg.hipp.y   = (1.f - hip.location.y) * size.y;
                        data.mRightLeg.kneep.x  = knee.location.x * size.x;
                        data.mRightLeg.kneep.y  = (1.f - knee.location.y) * size.y;
                        data.mRightLeg.anklep.x = ankle.location.x * size.x;
                        data.mRightLeg.anklep.y = (1.f - ankle.location.y) * size.y;
#endif
                        data.mRightLeg.hip_confidence = hip.confidence;
                        data.mRightLeg.knee_confidence = knee.confidence;
                        data.mRightLeg.ankle_confidence = ankle.confidence;
                    }
                                
                    // WAIST
                    {
                        VNRecognizedPoint *root = [allPts objectForKey:VNHumanBodyPoseObservationJointNameRoot];
#if NORMALIZE_RES
                        data.waistp.x          = root.location.x * aspect.x - aspect.y;
                        data.waistp.y          = (1.f - root.location.y);
#else
                        data.waistp.x          = root.location.x * size.x;
                        data.waistp.y          = (1.f - root.location.y) * size.y;
#endif
                        data.waist_confidence = root.confidence;
                    }
                                
                    // HEAD
                    {
                        VNRecognizedPoint *headNeck = [allPts objectForKey:VNHumanBodyPoseObservationJointNameNeck];
                        VNRecognizedPoint *headNose = [allPts objectForKey:VNHumanBodyPoseObservationJointNameNose];
                        VNRecognizedPoint *leftEar = [allPts objectForKey:VNHumanBodyPoseObservationJointNameLeftEar];
                        VNRecognizedPoint *leftEye = [allPts objectForKey:VNHumanBodyPoseObservationJointNameLeftEye];
                        VNRecognizedPoint *rightEar = [allPts objectForKey:VNHumanBodyPoseObservationJointNameRightEar];
                        VNRecognizedPoint *rightEye = [allPts objectForKey:VNHumanBodyPoseObservationJointNameRightEye];
                                 
#if NORMALIZE_RES
                        data.neckp.x       = headNeck.location.x * aspect.x - aspect.y;
                        data.neckp.y       = (1.f - headNeck.location.y);
                        data.nosep.x       = headNose.location.x * aspect.x - aspect.y;
                        data.nosep.y       = (1.f - headNose.location.y);
                        data.left_earp.x   = leftEar.location.x * aspect.x - aspect.y;
                        data.left_earp.y   = (1.f - leftEar.location.y);
                        data.left_eyep.x   = leftEye.location.x * aspect.x - aspect.y;
                        data.left_eyep.y   = (1.f - leftEye.location.y);
                        data.right_earp.x  = rightEar.location.x * aspect.x - aspect.y;
                        data.right_earp.y  = (1.f - rightEar.location.y);
                        data.right_eyep.x  = rightEye.location.x * aspect.x - aspect.y;
                        data.right_eyep.y  = (1.f - rightEye.location.y);
#else
                        data.neckp.x       = headNeck.location.x * size.x;
                        data.neckp.y       = (1.f - headNeck.location.y) * size.y;
                        data.nosep.x       = headNose.location.x * size.x;
                        data.nosep.y       = (1.f - headNose.location.y) * size.y;
                        data.left_earp.x   = leftEar.location.x * size.x;
                        data.left_earp.y   = (1.f - leftEar.location.y) * size.y;
                        data.left_eyep.x   = leftEye.location.x * size.x;
                        data.left_eyep.y   = (1.f - leftEye.location.y) * size.y;
                        data.right_earp.x  = rightEar.location.x * size.x;
                        data.right_earp.y  = (1.f - rightEar.location.y) * size.y;
                        data.right_eyep.x  = rightEye.location.x * size.x;
                        data.right_eyep.y  = (1.f - rightEye.location.y) * size.y;
#endif
                        data.neck_confidence      = headNeck.confidence;
                        data.nose_confidence      = headNose.confidence;
                        data.left_ear_confidence  = leftEar.confidence;
                        data.left_eye_confidence  = leftEye.confidence;
                        data.right_ear_confidence = rightEar.confidence;
                        data.right_eye_confidence = rightEye.confidence;
                    }
                    
                    tmpData.push_back(data);
                }
                _data = tmpData;
            } else _data.clear();
        }];
    }
    
    _rqDictionary = [[NSDictionary alloc] init];
}

void PoseTracker::release(){
    _data.clear();
    //[_rqDictionary release];
    if (@available(macOS 11.0, *)) {
        //[_mlRequestBody dealloc];
        //[_mlRequestBody release];
    }
}

std::vector<BodyData>& PoseTracker::process( CGImageRef surf ){
    if (@available(macOS 11.0, *)) {
        VNImageRequestHandler *bodyHandler  =
            [[VNImageRequestHandler alloc]
             initWithCGImage: surf
             options:         _rqDictionary];
    
        _rqBodyArray  = @[_mlRequestBody];
        [bodyHandler performRequests:_rqBodyArray error:nil];
        //[bodyHandler release];
        //[_rqBodyArray release];
        CGImageRelease(surf);
    }
    
    return _data;
}
