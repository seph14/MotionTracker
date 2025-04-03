//
//  MotionProcessor.cpp
//  Tracker
//
//  Created by SEPH LI on 07/02/2024.
//

#include "MotionProcessor.h"
#include "cinder/Log.h"
#include "cinder/app/App.h"
#include "Resources.h"
#include "cinder/CinderImGui.h"
#include "cinder/ImageIo.h"
#include "cinder/ip/Fill.h"
#include "cinder/Utilities.h"
#include "cinder/ip/Resize.h"
#include "cinder/ip/Blur.h"
#include "cinder/ip/Flip.h"
#include "cinder/Rand.h"
#include "cinder/FileWatcher.h"
#include "PoseHandler.h"
#include "DepthHandler.h"

#include <queue>
#include <unordered_set>
#include <algorithm>
#include <random>

#define IMAGE_TEST 0
#define USE_CALIB_TRANSFORM 0

// normalized value for how long does a skeleton is considered stable
static float DetectionThreshold = 0.5f;
//https://developer.apple.com/documentation/vision/vngeneratepersonsegmentationrequestqualitylevel?language=objc
// normalized ratio, used as the default fading ratio when a new skeleton is flashing in/out,
static float     FlashRatio      = .35f;

using namespace ci;
using namespace std;

#if IMAGE_TEST
ci::Surface8uRef mTestSurf;
#endif

MotionProcessorRef MotionProcessor::create(){
    return MotionProcessorRef(new MotionProcessor());
}

MotionProcessorRef MotionProcessor::create(const nlohmann::json& file){
    return MotionProcessorRef(new MotionProcessor(file));
}

MotionProcessor::MotionProcessor(const nlohmann::json& file){
    initParams();
    loadShaders();
    findCameras();
    printAvailableCameraNames();

    auto devices  = Capture::getDevices();
    auto& camData = file["cameras"];
    for(auto& cam : camData) {
        string name = cam["name"];
        CamTransform camTrans;
        
        for(int i = 0; i < devices.size(); i++) {
            auto device = devices[i];
            string uid  = device->getName() + "-" + device->getUniqueId();
            CI_LOG_I("try find match: " << uid << "::" << name);
            if(uid.find(name) != std::string::npos) {
                try {
                    camTrans.capture = Capture::create(
                        cam["resolution"][0], cam["resolution"][1], device );
                    camTrans.capture->start();
                    
                    // sometimes init multi cameras too fast makes the feed unstable
                    this_thread::sleep_for(chrono::milliseconds(100));
                    
                    camTrans.name    = uid;
                    devices.erase(devices.begin() + i);
                    CI_LOG_I("Adding camera " << uid);
                    i --;
                    break;
                } catch(const ci::Exception &ex) {
                    CI_LOG_EXCEPTION( "Failed to init capture ", ex );
                }
            }
        }
        
        // if camera is found
        if(camTrans.capture) {
            camTrans.translate  = vec3( cam["translate"][0], cam["translate"][1], cam["translate"][2] );
            camTrans.euler      = vec3( cam["euler"][0],     cam["euler"][1],     cam["euler"][2]     );
            camTrans.fov        = cam["fov"];
            camTrans.cutoff     = cam["cutoff"];
            camTrans.depthShift = vec2( cam["depthshift"][0], cam["depthshift"][1] );
            camTrans.numLimit   = cam["limit"];
            camTrans.flip       = cam["flip"];
            
            camTrans.updated    = false;
            mCamTransforms.push_back(camTrans);
        }
    }
    
    ivec2 size = ml::DepthHandler::getOutputSize();
    for(int i = 0; i < mCamTransforms.size(); i++){
        // 2 x depth input
        mDepthInput.push_back(Surface8u::create(ml::DepthHandler::getExpectedSize().x,
                                                ml::DepthHandler::getExpectedSize().y, false));
        auto depthChannel = Channel32f::create(size.x, size.y);
        mDepthFrame.push_back (depthChannel);
        mDepthTex.push_back   (gl::Texture2d::create(*depthChannel));
    }
    
    mCapTex.resize(mDepthInput.size());
    load(file);
    mProcessThd = shared_ptr<thread>(new thread(bind(&MotionProcessor::processBodyTracking, this)));
    
#if IMAGE_TEST
    mTestSurf = Surface8u::create(loadImage(app::loadAsset("test.jpg")));
#endif
}

MotionProcessor::MotionProcessor() {
    initParams();
    loadShaders();
    findCameras();

    auto devices = Capture::getDevices();
    int idx = 0;
    for(auto device : devices) {
        auto& setting = mCamNames[idx];
        
        // by default, search for facetime
        if(setting.first.find("FaceTime") != std::string::npos){
            try{
                CamTransform camTrans;
                camTrans.name    = setting.first + "-" + setting.second;
                camTrans.capture = Capture::create( 960, 540, devices[idx] );
                camTrans.capture->start();
                mCamTransforms.push_back(camTrans);
                CI_LOG_I("Adding camera " << camTrans.name);
            } catch(const ci::Exception &ex) {
                CI_LOG_EXCEPTION( "Failed to init capture ", ex );
            }
        }
        idx++;
    }
    
    mDepthInput.push_back(Surface8u::create(ml::DepthHandler::getExpectedSize().x,
                                            ml::DepthHandler::getExpectedSize().y, false));
    
    ivec2 size          = ml::DepthHandler::getOutputSize();
    auto depthChannel   = Channel32f::create(size.x, size.y);
    mDepthFrame.push_back(depthChannel);
    mDepthTex.push_back (gl::Texture2d::create(*depthChannel));
    mProcessThd         = shared_ptr<thread>(new thread(bind(&MotionProcessor::processBodyTracking, this)));
    mCapTex.resize(mDepthInput.size());
    
#if IMAGE_TEST
    mTestSurf = Surface8u::create(loadImage(app::loadAsset("test.jpg")));
#endif
}

void MotionProcessor::initParams() {
    mShouldQuit       = false;
    mDrawAllSkeleton  = false;
    mDrawSkeleton     = true;
    mAllFeedProcessed = false;
    mProcessRgb       = true;
    mNoRender         = false;
    mCamInited        = false;
    mMode             = RenderMode::ThreeD;
    
    mCam.lookAt(vec3(.0f, 4.5f, 8.f), vec3(.0f,0.f,.0f));
    mCamUi          = CameraUi(&mCam, app::getWindow());
    mFadeDuration   = vec2(2.f,1.5f);
    
    mProcessedFeedNum   = 0;
    mDistThreshold      = .2f;
    mMaxSkeleton        = 8;
    mActiveNum          = 0;
    mDetectedNum        = 0;
    mMeasurementNoise   = 1.00001f;
    mProcessNoise       = .01f;
    mSkeletonScale      = 2.5f;
    mFlatRange          = 3.f;
    mMLFrms             = 0;
    mFeedFrms           = 0;
    mUpdateTime         = 0.f;
    mMLFps              = 0.f;
    mFeedFps            = 0.f;
    mLastSampleTime     = 0.f;
    mWorldThreshold     = .4f;
    mDepthTargetIdx     = -1;
    resizeBodyCount();
    
    mBodyColor = {
        Color::hex(0xff0000), Color::hex(0x00ff00), Color::hex(0x0000ff),
        Color::hex(0x2797e6), Color::hex(0xe62263), Color::hex(0x583ac6),
        Color::hex(0xdcbc25), Color::hex(0x3fa478), Color::hex(0xba4222)
    };
    
    // debugging rendering font
    auto font = Font( "Arial Black", app::toPixels(24) );
    mTextureFont = gl::TextureFont::create( font );
}

void MotionProcessor::loadShaders() {
    
}

void MotionProcessor::findCameras(){
    auto devices = Capture::getDevices();
    for(auto device : devices){
        auto name = device->getName();
        mCamNames.push_back({name, device->getUniqueId()});
    }
}

void MotionProcessor::printAvailableCameraNames(){
    CI_LOG_I("------------------Available Cameras------------------");
    for(auto name : mCamNames) CI_LOG_I(name.first << "-" << name.second);
}

ci::Surface8uRef MotionProcessor::resizeImage(ci::Surface8uRef src,
                                              const ci::ivec2& targetSize) {
    auto newSurf = ip::resizeCopy(*src, src->getBounds(), vec2(targetSize.y, targetSize.x));
    auto inSurf  = Surface8u::create(targetSize.x, targetSize.y, false);
    
    auto pixelIter  = inSurf->getIter();
    while( pixelIter.line() ) {
        while( pixelIter.pixel() ) {
            auto col = newSurf.getPixel(ivec2(pixelIter.y(),
                                              targetSize.x - 1 - pixelIter.x()));
            pixelIter.r() = col.r;
            pixelIter.g() = col.g;
            pixelIter.b() = col.b;
        }
    }
    
    return inSurf;
}

void MotionProcessor::processBodyTracking(){
    ml::PoseTracker::init ();
    ml::DepthHandler::init();
    
    while(!mShouldQuit){
        if(mQueIn.size() == 0) {
            std::this_thread::yield();
            continue;;
        }
        
        CamFrame camFrame;
        mQueIn.pop_front_waiting(camFrame);

        MLFrame output;
    
        auto imgRef               = ml::PoseTracker::convetSurface(camFrame.frame);
        if(mDepthTargetIdx != camFrame.idx)
        mDepthFrame[camFrame.idx] = ml::DepthHandler::process(*camFrame.depth);
        output.bodyData           = ml::PoseTracker::process(imgRef);
        output.depthRange         = ml::DepthHandler::getDepthRange();

        // assign camera and depth
        for(auto& body : output.bodyData) {
            body.camIdx = camFrame.idx;
            body.assignDepth(mDepthFrame[camFrame.idx], mCamTransforms[camFrame.idx].depthShift);
        }
        
        output.idx      = camFrame.idx;
        output.depth    = mDepthFrame[camFrame.idx];
        mQueOut.push_back(output);
        std::this_thread::yield();
    }
    
    ml::DepthHandler::release();
    ml::PoseTracker::release();
    CI_LOG_I("ML model released");
}

MotionProcessor::~MotionProcessor(){
    mShouldQuit = true;
#if !TRACKING_DEBUG
    for(auto& cam : mCamTransforms)
        cam.capture->stop();
#endif
    mProcessThd->join();
    mProcessThd.reset();
}

void MotionProcessor::remapBodyIdx(const int& oldIdx, const int& newIdx) {
    mTargetBody[newIdx] = mTargetBody[oldIdx];
    
    auto& oldData       = mTargetBody[oldIdx];
    oldData.idx         = -1;
    oldData.overlapRef  = -1;
    oldData.camIdx      = -1;
    oldData.fadeIn      = 0.f;
    oldData.fadeOut     = 1.f;
    oldData.updated     = false;
}

void MotionProcessor::resetBodyIdx(){
    queue<int> resetIdx, activeIdx;
    if(mRemovedIdx.size() >= 32)
        mRemovedIdx.clear();
    for (int i = 0; i < mMaxSkeleton; i++) {
        auto& data = mTargetBody[i];
        if(data.idx >= 0 && data.fadeOut < 0.f) {
            CI_LOG_D("Reset skeleton from scale: " << i << "/" << data.fadeOut << "/" << data.idx);
            mActiveNum      = glm::max(0, mActiveNum - 1);
            mRemovedIdx.push_back(data.idx);
            data.idx        = -1;
            data.overlapRef = -1;
            data.camIdx     = -1;
            data.fadeIn     = 0.f;
            data.fadeOut    = 1.f;
            resetIdx.push(i);
        }
    }

    if(resetIdx.size() > 0) {
        for (int i = 0; i < mMaxSkeleton; i++) {
            if(mTargetBody[i].idx >= 0)
                activeIdx.push(i);
        }
    }

    while (activeIdx.size() > 0) {
        int headIdx = activeIdx.front();
        activeIdx.pop();
        
        for (int ii = 0; ii < resetIdx.size(); ii++) {
            int deactIdx = resetIdx.front();
            if (deactIdx < headIdx) {
                CI_LOG_D("remap " << headIdx << " -> " << deactIdx);
                remapBodyIdx(headIdx, deactIdx);
                resetIdx.pop();
                break;
            }
        }
    }
    
    std::sort(mTargetBody.begin(), mTargetBody.end(), [](const SkeletonData& a, const SkeletonData& b){
        return a.fadeIn > b.fadeIn;
    });
}

void MotionProcessor::resizeBodyCount(){
    int oldCnt = (int)mTargetBody.size();
    if(mMaxSkeleton == oldCnt) return;
    
    mTargetBody.resize(mMaxSkeleton);
    if(mMaxSkeleton > oldCnt){
        for(int i = oldCnt; i < mMaxSkeleton; i++){
            auto& data = mTargetBody[i];
            data.idx        = -1;
            data.camIdx     = -1;
            data.overlapRef = -1;
            data.fadeIn     = 0.f;
            data.fadeOut    = 1.f;
        }
    }
}

const size_t MotionProcessor::availableSkeletonCnt() {
    uint32_t i = 0;
    for(auto& body : mTargetBody)
        if(isSkeletonStable(body)) i++;
    return i;
}

std::vector<ci::vec3> MotionProcessor::skeletonData(const size_t& idx) {
    auto& body = mTargetBody[idx];
    float val  = glm::min(1.f, glm::min(body.fadeIn, body.fadeOut));
    
    return {
        // meta data
        vec3(body.idx, 
             glm::max(0.f, (val - DetectionThreshold) / (1.f - DetectionThreshold)),
             isSkeletonStable(body) ? 1.f : 0.f),
        
        // left leg
        body.wpos[0],
        body.wpos[1],
        body.wpos[2],
        // right leg
        body.wpos[3],
        body.wpos[4],
        body.wpos[5],
        // body
        body.wpos[6],
        body.wpos[7],
        body.wpos[8],
        body.wpos[9],
        body.wpos[10],
        body.wpos[11],
        body.wpos[12],
        // left arm
        body.wpos[13],
        body.wpos[14],
        body.wpos[15],
        // right arm
        body.wpos[16],
        body.wpos[17],
        body.wpos[18]
    };
}

void MotionProcessor::setupCamera(CamTransform& camTrans, uint8_t camIdx) {
    if(camTrans.aspectRatio < 0.f && camTrans.frame != nullptr){
        camTrans.aspectRatio = camTrans.frame->getAspectRatio();
        
        auto idx     = camTrans.name.find("-");
        auto camname = camTrans.name.substr(0, idx);
        
        auto camdataPath = app::getAssetPath("cameradata.json");
        if(camdataPath.empty()){
            // generate dummy coeffs
            camTrans.data.generateDummy();
        } else {
            auto camData  = nlohmann::json::parse(loadString(loadFile(camdataPath)));
            if(camData["camera"].find(camname) != camData["camera"].end()) {
                camTrans.data.load  (camData["camera"][camname]);
            } else {
                // generate dummy coeffs
                camTrans.data.generateDummy();
            }
        }
        
        camTrans.cam.setFarClip     (5.f);
        camTrans.cam.setAspectRatio (camTrans.aspectRatio);
        camTrans.cam.setFovHorizontal(camTrans.fov);
        camTrans.cam.setEyePoint    (vec3(0.f));
        
        // main camera
#if USE_CALIB_TRANSFORM
        if(camIdx == 0) {
#endif
            camTrans.transform.from     (camTrans.translate, camTrans.euler);
            camTrans.cam.setOrientation (glm::quat(glm::radians(camTrans.euler)));
            camTrans.cam.setEyePoint    (camTrans.translate);
            camTrans.worldMatrix = camTrans.transform.to_touch();
#if  USE_CALIB_TRANSFORM
        // sub camera, load transform
        } else {
            camTrans.transform.setFromJson(camData["transform"]);
        }
#endif
        CI_LOG_I(camTrans.name << " inited");
    }
}

const ci::vec3 MotionProcessor::unproject(const ci::vec3 &p, const CamTransform &cam) {
    // add capability to scale view space positions
    return cam.worldMatrix * vec4(vec3(-mSkeletonScale, mSkeletonScale, 1.f) * p, 1.f);
}

void MotionProcessor::mapBody( SkeletonData& body, const CamTransform& camTrans ){
    // left leg
    body.wpos[0]  = unproject(body.data.mLeftLeg.ankle.pos(), camTrans);
    body.wpos[1]  = unproject(body.data.mLeftLeg.hip.pos(),   camTrans);
    body.wpos[2]  = unproject(body.data.mLeftLeg.knee.pos(),  camTrans);
    // right leg
    body.wpos[3]  = unproject(body.data.mRightLeg.ankle.pos(),camTrans);
    body.wpos[4]  = unproject(body.data.mRightLeg.hip.pos(),  camTrans);
    body.wpos[5]  = unproject(body.data.mRightLeg.knee.pos(), camTrans);
    // body
    body.wpos[6]  = unproject(body.data.waist.pos(),    camTrans);
    body.wpos[7]  = unproject(body.data.neck.pos(),     camTrans);
    body.wpos[8]  = unproject(body.data.nose.pos(),     camTrans);
    body.wpos[9]  = unproject(body.data.left_eye.pos(), camTrans);
    body.wpos[10] = unproject(body.data.left_ear.pos(), camTrans);
    body.wpos[11] = unproject(body.data.right_eye.pos(),camTrans);
    body.wpos[12] = unproject(body.data.right_ear.pos(),camTrans);
    // left arm
    body.wpos[13] = unproject(body.data.mLeftArm.shoulder.pos(), camTrans);
    body.wpos[14] = unproject(body.data.mLeftArm.elbow.pos(),    camTrans);
    body.wpos[15] = unproject(body.data.mLeftArm.wrist.pos(),    camTrans);
    // right arm
    body.wpos[16] = unproject(body.data.mRightArm.shoulder.pos(),camTrans);
    body.wpos[17] = unproject(body.data.mRightArm.elbow.pos(),   camTrans);
    body.wpos[18] = unproject(body.data.mRightArm.wrist.pos(),   camTrans);
}

void MotionProcessor::drawRGBComposition() {
    gl::ScopedViewport    scpView  (mRgbBound);
    gl::ScopedMatrices    scpMat;
    gl::setMatricesWindow(mRgbBound, false);
    for(uint32_t i = 0; i < mCamTransforms.size(); i++) {
        auto& bound = mCamTransforms[i].viewRange;
        gl::draw(mCapTex[i], Rectf(bound.x,bound.y,bound.z,bound.w));
    }
}

const bool MotionProcessor::isSkeletonStable(const SkeletonData& data){
    return ( data.fadeIn > DetectionThreshold && data.fadeOut > 0.f && data.overlapRef < 0);
}

bool MotionProcessor::update(const float& dt, const float& time) {
    bool frameUpdated = false;
    
    {
        uint8_t camIdx = 0;
        for(auto& trans : mCamTransforms) {
            if(trans.capture && !trans.updated && trans.capture->checkNewFrame()) {
#if IMAGE_TEST
                trans.frame  = mTestSurf;
#else
                trans.frame  = trans.capture->getSurface();
#endif
                if(trans.flip){
                    auto newsurf = ci::Surface8u::create(trans.frame->getWidth(), trans.frame->getHeight(), false);
                    ip::flipVertical(*trans.frame, newsurf.get());
                    trans.frame = newsurf;
                }
                
                setupCamera(trans, camIdx);
                trans.updated |= trans.frame != nullptr;
                // prepare depth feed
                ip::resize(*trans.frame, mDepthInput[camIdx].get());

                // push the new cam frame into queue
                mQueIn.push_back(CamFrame{
                    camIdx, trans.numLimit, trans.frame, mDepthInput[camIdx]
                });
                CI_LOG_D("Pushing camera " << (int)camIdx);
                mFeedFrms ++;

#if BODY_TEST
                if(mProcessRgb){
                    if(mCapTex[camIdx]){
                        mCapTex[camIdx]->update(*trans.frame);
                    } else {
                        mCapTex[camIdx] = gl::Texture2d::create(*trans.frame,
                            gl::Texture2d::Format().loadTopDown());
                    }
                }
#endif
            }
            camIdx ++;
        }
        
        if(mRgbBound.x == 0) {
            bool allUpdated = true;
            for(auto& trans : mCamTransforms)
                if(!trans.updated) allUpdated = false;
            
            if(allUpdated) {
                // handle stereo transform binding
                if(!mCamInited && mCamTransforms.size() > 1) {
                    auto& transformPivot = mCamTransforms[0].transform;
                    auto& camTrans       = mCamTransforms[1];
                    auto subPivot        = camTrans.transform;
                    camTrans.transform   = transformPivot.compose_with(subPivot);
                    camTrans.translate   = vec3(camTrans.transform.t(0),
                                                camTrans.transform.t(1),
                                                camTrans.transform.t(2));
                    camTrans.cam.setOrientation (glm::quat_cast(camTrans.transform.rotation()));
                    camTrans.cam.setEyePoint    (camTrans.translate);
                    camTrans.worldMatrix = camTrans.transform.to_touch();
                    mCamInited = true;
                }
                
                // assign rgb bound
                for(auto& cam : mCamTransforms) {
                    mRgbBound.x += cam.frame->getWidth();
                    mRgbBound.y  = glm::max( mRgbBound.y, cam.frame->getHeight() );
                }
                
                // init cam frames
                int offset = 0;
                for(int i = 0; i < mCamTransforms.size(); i++) {
                    auto& camTrans      = mCamTransforms[i];
                    camTrans.viewRange  = ivec4(offset, 0,
                                                offset + camTrans.frame->getWidth(), mRgbBound.y);
                    CI_LOG_I("View " << i << " " << camTrans.viewRange);
                    offset += camTrans.frame->getWidth();
                }
                
                CI_LOG_I("Initializing camera buffers");
            }
        }
    }
    
    while(mQueOut.size() > 0 && mProcessedFeedNum < mCamTransforms.size()) {
        MLFrame mlFrame;
        mQueOut.pop_front_waiting(mlFrame);
        CI_LOG_D("Process camera " << (int)mlFrame.idx << "-" << mProcessedFeedNum);
        
        // update depth tex
        mDepthTex[mlFrame.idx]->update(*mlFrame.depth);

        // free this camera for next frame
        mProcessedFeedNum ++;
        
        // update segmentation tex
        if(!mAllFeedProcessed){
            if(mProcessedFeedNum >= mCamTransforms.size() && !mAllFeedProcessed)
                mAllFeedProcessed = true;
            if(mAllFeedProcessed) {
                for(auto& cam : mCamTransforms) cam.updated = false;
            }
        } else
            mCamTransforms[mlFrame.idx].updated = false;
        mCamTransforms[mlFrame.idx].depthRange = mlFrame.depthRange;
        
        // update body tracking
        for(auto data : mlFrame.bodyData)
            if(data.isvalid(mCamTransforms[data.camIdx].cutoff))
                mTmpData.push_back(data);
    }
    
    if(mProcessedFeedNum >= mCamTransforms.size()){
        // reset processed num
        mProcessedFeedNum = 0;
        mMLFrms ++;
        mDetectedNum = (int32_t)mTmpData.size();
        CI_LOG_D("Process full data frame " << mDetectedNum);

        // recycle body data
        resetBodyIdx();
        
        float processDt = time - mLastSampleTime;
        mLastSampleTime = time;
        
        // map bodies by their distance
        if(mTmpData.size() > 0) {
            // shuffle body order
            auto rng = std::default_random_engine {};
            std::shuffle(mTmpData.begin(), mTmpData.end(), rng);
            
            unordered_set<int> selectedIdx;
            int updatedNum = 0;
            
            for(auto& data : mTargetBody ){
                // reset
                data.updated = false;
                if(data.idx < 0) continue;
                
                float thres = FLT_MAX - 1.f;
                int   idx   = -1;
                vec3  piv   = data.data.waistp; // compare 2d space
                
                for(int i = 0; i < mTmpData.size(); i++) {
                    if(selectedIdx.find(i) != selectedIdx.end()) continue;
                    auto& body  = mTmpData[i];
                    if(body.camIdx != data.camIdx) continue;
                    
                    // todo: should use resolution of the corresponding camera
                    float dist  = glm::length2(vec2(piv - body.waistp)) / 1280.f / 1280.f;
                    if(dist < thres) {
                        idx     = i;
                        thres   = dist;
                    }
                }
                
                // update this skeleton
                if(idx >= 0 && thres < mDistThreshold * mDistThreshold) {
                    selectedIdx.insert(idx);
                    data.updated = true;
                    
                    // kalman filter applies on world position
                    data.data.update  (mTmpData[idx]);
                    data.data.mapWorld(processDt, mCamTransforms[mTmpData[idx].camIdx].data);
                    // unproject from 2d space to camera view to world space
                    mapBody(data, mCamTransforms[mTmpData[idx].camIdx]);

                    data.fadeIn   = glm::min(data.fadeIn + dt / mFadeDuration.x, 100.f);
                    data.fadeOut  = (data.fadeIn >= 1.f) ? 1.f : FlashRatio;
                    updatedNum++;
                } //else CI_LOG_I(idx << ":: " << thres << " / " << (mDistThreshold * mDistThreshold));
            }
            
            updatedNum = 0;
            vector<uint32_t> camSlots(4, 0);
            for(auto& b : mTargetBody){
                if(b.idx >= 0) {
                    updatedNum ++;
                    camSlots[b.camIdx] ++;
                }
            }
            
            // if we have unused skeleton and there are empty slots in tracked bodies
            if(selectedIdx.size() < mTmpData.size() && updatedNum < mMaxSkeleton) {
                // find available idx
                vector<int32_t> takenIndices;
                queue<int32_t> availableIndices;
                for(auto& data : mTargetBody)
                    if(data.idx >= 0) takenIndices.push_back(data.idx);
                
                int32_t emptyCnt = (int32_t)(mTmpData.size() - selectedIdx.size());
                for(int32_t eid = 0; eid < 128; eid ++){ // we are using byte in unity
                    if(std::find(takenIndices.begin(), takenIndices.end(), eid) == takenIndices.end() &&
                       std::find(mRemovedIdx.begin(), mRemovedIdx.end(), eid) == 
                        mRemovedIdx.end()) {
                        
                        availableIndices.push(eid);
                        if(availableIndices.size() >= emptyCnt) break;
                    }
                }
                
                if(availableIndices.size() > 0){
                    for(auto& data : mTargetBody){
                        if(data.idx >= 0/*&& isSkeletonStable(data)*/) continue;
                        
                        bool assigned = false;
                        for(int i = 0; i < mTmpData.size(); i++) {
                            if(selectedIdx.find(i) != selectedIdx.end()) continue;
                            
                            auto& tmpData = mTmpData[i];
                            if(camSlots[tmpData.camIdx] >= mCamTransforms[tmpData.camIdx].numLimit) continue;;
                            
                            data.data.reset(tmpData,
                                            mCamTransforms[tmpData.camIdx].data);
                            mapBody(data, mCamTransforms[tmpData.camIdx]);
                            
                            data.updated = true;
                            data.idx     = availableIndices.front();
                            data.camIdx  = tmpData.camIdx;
                            data.fadeIn  = dt / mFadeDuration.x;
                            data.fadeOut = FlashRatio;
                            
                            camSlots[tmpData.camIdx] ++;
                            selectedIdx.insert(i);
                            availableIndices.pop();
                            assigned = true;
                            CI_LOG_I("assign new body: " << i << "-" << data.idx << ":" << (int)data.camIdx);
                            if(availableIndices.size() == 0) break;
                            if(selectedIdx.size() == mTmpData.size()) break;
                        }
                        
                        if(assigned) mActiveNum ++;
                        if(availableIndices.size() == 0) break;
                        if(selectedIdx.size() == mTmpData.size()) break;
                    }
                }
            }
        }
        
        // reset reference count
        // fade out non-used body data
        for(auto& body : mTargetBody) {
            body.overlapRef = -1;
            body.fadeOut   -= dt / mFadeDuration.y;
        }
        
        // simple logic to handle where skeleton overlap on multi cameras
        // if mapped world position is within a threshold, consider the skeleton is the same one
        const float dThres = mWorldThreshold * mWorldThreshold;
        for(uint32_t i = 0; i < mMaxSkeleton; i++ ){
            auto& currBody = mTargetBody[i];
            if(currBody.idx < 0 || currBody.fadeOut <= 0.f || currBody.overlapRef >= 0)
                continue;
            
            vec3 piv    = currBody.wpos[6] * vec3(1.f,0.f,0.f);
            float   mind= FLT_MAX;
            int32_t ref = -1;
            
            for(uint32_t j = i+1; j < mMaxSkeleton; j++){
                auto& otherBody = mTargetBody[j];
                if(otherBody.idx < 0 || currBody.fadeOut <= 0.f ||
                   otherBody.overlapRef >= 0 || otherBody.camIdx == currBody.camIdx)
                    continue;
                
                vec3 pp = otherBody.wpos[6] * vec3(1.f,0.f,0.f);
                float d = glm::length2(piv - pp);
                if(d < dThres && d < mind) {
                    mind = d;
                    ref  = j;
                }
            }
            
            // make the one exists longer as the main skeleton,
            // and hide the other one
            if(ref >= 0) {
                auto& other = mTargetBody[ref];
                if(currBody.fadeIn > other.fadeIn || currBody.fadeOut < other.fadeOut) {
                    other.overlapRef = i;
                } else currBody.overlapRef = ref;
            }
        }
        
        mBodyData    = mTmpData;
        mTmpData.clear();
        frameUpdated = true;
    }
    
    return mAllFeedProcessed && frameUpdated;
}

// calculate a linear regression that maps relative depth estimation to metric values
ci::vec2 MotionProcessor::calculateDepthShift(const std::vector<float>& rawDepth,
                                              const std::vector<float>& targetDepth){
    vec2 param;
    
    const size_t n = rawDepth.size();
    if (n < 2) {
        CI_LOG_E("Require at least 2 points for calculation.");
        return param;
    }

    float sum_x = 0.f, sum_y = 0.f, sum_xy = 0.f, sum_x2 = 0.f;
    
    for(uint32_t i = 0; i < n; i++){
        auto raw    = rawDepth[i];
        auto target = targetDepth[i];
        
        sum_x   += 1.f - raw;
        sum_y   += target;
        sum_xy  += target * (1.f - raw);
        sum_x2  += (1.f - raw) * (1.f - raw);
    }
    
    const float denominator = static_cast<float>(n) * sum_x2 - sum_x * sum_x;
    if (std::abs(denominator) < 1e-10) {
        CI_LOG_E("Can not calculate, vertical line or too many overlapped points");
        return param;
    }

    // slope
    param.y = (static_cast<float>(n) * sum_xy - sum_x * sum_y) / denominator;
    //intercept
    param.x = (sum_y - param.x * sum_x) / static_cast<float>(n);
    return param;
}

void MotionProcessor::drawSkeleton3d(const std::array<ci::vec3,19>& wpos) {
    static const float scl = 1.f;
    /*
     // left leg
     0 -  left ankle
     1 -  left hip
     2 -  left knee
     // right leg
     3 -  right ankle
     4 -  right hip
     5 -  right knee
     // body
     6 -  waist
     7 -  neck
     8 -  nose
     9 -  left eye
     10 - left ear
     11 - right eye
     12 - right ear
     // left arm
     13 - left shoulder
     14 - left elbow
     15 - left wrist
     // right arm
     16 - right shoulder
     17 - right elbow
     18 - right wrist
     */
    
    gl::drawCube(scl * wpos[13], vec3(.125f/6.f));
    gl::drawCube(scl * wpos[14], vec3(.125f/6.f));
    gl::drawCube(scl * wpos[15], vec3(.125f/6.f));
    gl::drawLine(scl * wpos[13], scl * wpos[14]);
    gl::drawLine(scl * wpos[15], scl * wpos[14]);
    
    gl::drawCube(scl * wpos[16], vec3(.125f/6.f));
    gl::drawCube(scl * wpos[17], vec3(.125f/6.f));
    gl::drawCube(scl * wpos[18], vec3(.125f/6.f));
    gl::drawLine(scl * wpos[16], scl * wpos[17]);
    gl::drawLine(scl * wpos[18], scl * wpos[17]);
 
    gl::drawCube(scl * wpos[0], vec3(.125f/6.f));
    gl::drawCube(scl * wpos[1], vec3(.125f/6.f));
    gl::drawCube(scl * wpos[2], vec3(.125f/6.f));
    gl::drawLine(scl * wpos[0], scl * wpos[2]);
    gl::drawLine(scl * wpos[1], scl * wpos[2]);
 
    gl::drawCube(scl * wpos[3], vec3(.125f/6.f));
    gl::drawCube(scl * wpos[4], vec3(.125f/6.f));
    gl::drawCube(scl * wpos[5], vec3(.125f/6.f));
    gl::drawLine(scl * wpos[3], scl * wpos[5]);
    gl::drawLine(scl * wpos[4], scl * wpos[5]);

    {
        gl::drawCube(scl * wpos[6], vec3(.125f/6.f));
        gl::drawLine(scl * wpos[1], scl * wpos[6]);
        gl::drawLine(scl * wpos[4], scl * wpos[6]);
    }
    
    {
        gl::drawCube(scl * wpos[7],  vec3(.125f/6.f));
        gl::drawCube(scl * wpos[8],  vec3(.125f/6.f));
        gl::drawCube(scl * wpos[9],  vec3(.075f/6.f));
        gl::drawCube(scl * wpos[10], vec3(.075f/6.f));
        gl::drawCube(scl * wpos[11], vec3(.075f/6.f));
        gl::drawCube(scl * wpos[12], vec3(.075f/6.f));
        
        gl::drawLine(scl * wpos[7], scl * wpos[8]);
        gl::drawLine(scl * wpos[9], scl * wpos[11]);
        
        gl::drawLine(scl * wpos[7],  scl * wpos[6]);
        gl::drawLine(scl * wpos[13], scl * wpos[7]);
        gl::drawLine(scl * wpos[7],  scl * wpos[16]);
    }
}

void MotionProcessor::draw(const ci::Rectf& bound) {
    if(mNoRender) return;
    
#if BODY_TEST
    if(mMode != RenderMode::ThreeD && mCapTex.size() > 0 && mtxBody.try_lock()){
        gl::viewport         (app::toPixels(app::getWindowSize()));
        gl::setMatricesWindow(app::toPixels(app::getWindowSize()), mMode == RenderMode::RGB );
        gl::clear            (Color::black());
        
        ivec2 offset = ivec2();
        ivec2 size   = app::toPixels(app::getWindowSize()) / ivec2((int32_t)mCamTransforms.size(), 1);
        if(mMode == RenderMode::Depth) {
            for(auto& depth : mDepthTex){
                gl::draw(depth, Rectf(offset.x, offset.y + size.y, offset.x + size.x, offset.y));
                offset.x += size.x;
            }
        
            {
                gl::ScopedColor scpColor(Color::hex(0xff0000));
                gl::setMatricesWindow(app::toPixels(app::getWindowSize()), true );
                for(uint32_t i = 0; i < mScrDepthPnts.size(); i++){
                    auto& p = mScrDepthPnts[i];
                    gl::drawSolidRect(Rectf(p.x-2.f,p.y-2.f,p.x+2.f,p.y+2.f));
                    mTextureFont->drawString(toString(i), p);
                }
            }
            
        } else if(mMode == RenderMode::RGB) {
            for(auto& rgb : mCapTex){
                gl::draw(rgb, Rectf(offset.x, offset.y, offset.x + size.x, offset.y + size.y));
                offset.x += size.x;
            }
        }
        
        if(mDrawSkeleton && mCapTex.size() > 0) {
            const float aspect = mCapTex[0]->getAspectRatio();
            if(mDrawAllSkeleton) {
                for(auto& body : mBodyData)
                    body.draw(aspect, mCamTransforms[body.camIdx].cutoff);
            } else {
                int idx = 0;
                for(auto& data : mTargetBody){
                    if(data.idx < 0) continue;
                    gl::ScopedColor scpColor( Color::black().lerp(
                       glm::min(1.f, glm::min(data.fadeIn, data.fadeOut)), mBodyColor[idx++]) );
                    data.data.drawNoColor(aspect);
                }
            }
        }
        
        mtxBody.unlock();
    } else {
        gl::viewport    (app::toPixels(app::getWindowSize()));
        gl::setMatrices (mCam);
        gl::clear       (Color::gray(.2f));

        {
            gl::ScopedColor scpColor(Color::hex(0x00ff00));
            gl::drawLine(vec3(.0f), vec3(0.f,3.f,0.f));
        }
        
        {
            gl::ScopedColor scpColor(Color::hex(0xc6971b));
            gl::drawLine(vec3(bound.getX1(),0.f,bound.getY1()),
                         vec3(bound.getX2(),0.f,bound.getY1()));
            gl::drawLine(vec3(bound.getX1(),0.f,bound.getY2()),
                         vec3(bound.getX2(),0.f,bound.getY2()));
            gl::drawLine(vec3(bound.getX1(),0.f,bound.getY1()),
                         vec3(bound.getX1(),0.f,bound.getY2()));
            gl::drawLine(vec3(bound.getX2(),0.f,bound.getY1()),
                         vec3(bound.getX2(),0.f,bound.getY2()));
        }
        
        if(mDrawSkeleton && mtxBody.try_lock()){
            static const float scl = .2f;
            
            {
                gl::ScopedColor scpColor(Color::white());
                for(auto& trans : mCamTransforms)
                    gl::drawFrustum(trans.cam);
            }
            
            if(mDrawAllSkeleton) {
                gl::ScopedColor scpColor(Color::hex(0x23cc00));
                for(auto& body : mBodyData)
                    body.draw3D(scl * vec3(mFlatRange, mCamTransforms[body.camIdx].depthShift),
                                scl * 2.25f * mFlatRange);
            } else {
                for(auto& data : mTargetBody){
                    if(data.idx < 0 || !isSkeletonStable(data)) continue;
                    gl::ScopedColor scpColor( Color::black().lerp(glm::min(1.f, glm::min(data.fadeIn, data.fadeOut)),
                                                                  mBodyColor[data.idx % mBodyColor.size()]) );
                    drawSkeleton3d(data.wpos);
                }
            }
            mtxBody.unlock();
            
            {
                gl::setMatricesWindow(app::toPixels(app::getWindowSize()));
                gl::ScopedBlendAlpha scpAlp;
                gl::ScopedColor scpColor(Color::white());
                uint32_t camIdx = 0;
                for(auto& trans : mCamTransforms) {
                    vec2 p = mCam.worldToScreen(trans.cam.getEyePoint(),
                                                app::toPixels(app::getWindowWidth()), app::toPixels(app::getWindowHeight()));
                    //gl::drawString(trans.name, p);
                    mTextureFont->drawString(toString(camIdx++), p);
                }
                
                for(auto& data : mTargetBody){
                    if(data.idx < 0 || !isSkeletonStable(data)) continue;
                    vec2 p = mCam.worldToScreen(data.wpos[6],
                        app::toPixels(app::getWindowWidth()), app::toPixels(app::getWindowHeight()));
                    vec3 wp = glm::round(10.f * data.wpos[6]) / 10.f;
                    mTextureFont->drawString(toString(wp), p);
                }
            }
        }
    }
#endif
}

void MotionProcessor::addDepthPoint(const ci::vec2& pnt){
    if(mMode == RenderMode::Depth) {
        int8_t newCamIdx = (int8_t)glm::floor(pnt.x * mCamTransforms.size());
        if(mDepthTargetIdx < 0 || mDepthTargetIdx == newCamIdx ){
            mDepthTargetIdx = newCamIdx;
            float xx = glm::fract(pnt.x * (float)mCamTransforms.size());
            ivec2 pp = glm::round((vec2)mDepthFrame[mDepthTargetIdx]->getSize()
                                  * vec2(xx, pnt.y));
            mRawDepth.push_back (*mDepthFrame[mDepthTargetIdx]->getData(pp));
            mRealDepth.push_back(0.f);
            mScrDepthPnts.push_back(glm::round(pnt *
                                    (vec2)app::toPixels(app::getWindowSize())));
        }
    }
}

void MotionProcessor::drawDepthUi() {
    // actions for depth estimation param edit
    if(mMode == RenderMode::Depth) {
        static const string name = "Camera Depth Edit";
        ImGui::Text("%s", name.c_str());
        for(uint32_t i = 0; i < mRawDepth.size(); i++){
            ImGui::ScopedId scpId(i);
            ImGui::Text("%s", toString(i).c_str());
            ImGui::InputFloat("Raw", &mRawDepth[i]);
            ImGui::SameLine  ();
            ImGui::InputFloat("Real", &mRealDepth[i]);
        }
        
        if(mRawDepth.size() > 2 && mDepthTargetIdx >= 0){
            if(ImGui::Button("Calculate")){
                mCamTransforms[mDepthTargetIdx].depthShift
                    = calculateDepthShift(mRawDepth, mRealDepth);
            }
            
            if(ImGui::Button("Reset")){
                mDepthTargetIdx = -1;
                mRawDepth.clear();
                mRealDepth.clear();
                mScrDepthPnts.clear();
            }
        }
    }
}

void MotionProcessor::drawUi() {
    ImGui::ScopedId scpId("motion");
    ImGui::Checkbox("No Render", &mNoRender);
    ImGui::Text       ("Motion Handler");
    ImGui::Checkbox   ("Draw Skeleton",        &mDrawSkeleton);
    ImGui::Checkbox   ("Draw All Skeleton",    &mDrawAllSkeleton);
    ImGui::Checkbox   ("Process RGB",          (bool*)&mProcessRgb);
    static const vector<string> modes = {"3D", "Depth", "RGB"};
    if(ImGui::Combo      ("Render Mode", (int*)&mMode, modes)){
        if(mMode == RenderMode::ThreeD) mCamUi.enable();
        else mCamUi.enable(false);
    }
    
    // we have no cameras added, list available cameras
    if(mCamTransforms.size() == 0) {
        uint32_t idx = 0;
        
        ImGui::Dummy(ivec2(0,5));
        ImGui::Text ("Available Cameras");
        for(auto& cam : mCamNames) {
            auto str = cam.first + "-" + cam.second;
            ImGui::ScopedId scpId(idx++);
            ImGui::InputText("", &str);
        }
        return;
    }
    
    ImGui::Dummy(ivec2(0,5));
    string active  = "Active Skeleton: " + toString(mActiveNum);
    ImGui::Text("%s", active.c_str());
    string detected = "Detected Skeleton: " + toString(mDetectedNum);
    ImGui::Text("%s", detected.c_str());
    
    float ctime = (float)app::getElapsedSeconds();
    if(ctime - mUpdateTime > 1.f){
        mMLFps      = float(mMLFrms) / (ctime - mUpdateTime);
        mFeedFps    = float(mFeedFrms) / (ctime - mUpdateTime);
        mUpdateTime = ctime;
        mMLFrms     = 0;
        mFeedFrms   = 0;
    }
    string mlfps = "ML Fps: " + toString(glm::round(100.f * mMLFps) / 100.f);
    ImGui::Text("%s", mlfps.c_str());
    string fefps = "Feed Fps: " + toString(glm::round(100.f * mFeedFps) / 100.f);
    ImGui::Text("%s", fefps.c_str());
    
    static bool noiseUpdated = false;
    noiseUpdated |= ImGui::SliderFloat("Process noise",     &mProcessNoise,     .0f, 1.f);
    noiseUpdated |= ImGui::SliderFloat("Measurement noise", &mMeasurementNoise, .0f, 15.f);
    if(noiseUpdated){
        for(auto& body : mBodyData)
            body.setNoise(mMeasurementNoise, mProcessNoise);
        noiseUpdated = false;
    }
    
    ImGui::DragFloat  ("Camera Space Threshold",&mDistThreshold,  .01f, .05f, 1.f);
    ImGui::DragFloat  ("World Dist Threshold",  &mWorldThreshold);
    
    ImGui::DragFloat("Flat Scalar",       &mFlatRange,          .01f, .1f,  6.f);
    ImGui::DragFloat("Unprojected Scalar",&mSkeletonScale,      .01f, 1.f,  8.f);
    ImGui::DragFloat("FadeIn Duration",   &mFadeDuration.x,     .01f, .1f,  2.f);
    ImGui::DragFloat("FadeOut Duration",  &mFadeDuration.y,     .01f, .1f,  2.f);
    ImGui::DragFloat("Valid Threshold",   &DetectionThreshold,  .01f, .1f, .95f);
    
    if(ImGui::SliderInt("Max Skeleton", &mMaxSkeleton, 1, 16))
        resizeBodyCount();
    
    int cnt = 0;
    for(int i = 0; i < mMaxSkeleton; i++) {
        auto& data = mTargetBody[i];
        if(!isSkeletonStable(data)) continue;
        cnt++;
        string str = toString(i) + " unique id: "
                   + toString(data.idx) + " / "
                   + toString(glm::round(100.f * glm::min(1.f, data.fadeIn))) + "::"
                   + toString(glm::round(100.f * data.fadeOut)) + "%"
                   + toString((int)data.camIdx);
        if(data.overlapRef >= 0) str += " ref:" + toString(data.overlapRef);
        ImGui::Text("%s", str.c_str());
        if(cnt == 8){
            ImGui::Separator();
            ImGui::Dummy(ivec2(0,5));
        }
    }
    
#if BODY_TEST
    ImGui::Text("RGB Frames");
    uint32_t cidx = 0;
    for(auto& frm : mCapTex) {
        if(frm){
            ImGui::ScopedId scpId(cidx);
            auto& trans = mCamTransforms[cidx];
            ImGui::Text("%s", trans.name.c_str());
            if(cidx == mDepthTargetIdx) drawDepthUi();
            
            ImGui::Checkbox("Flip Feed",          &trans.flip);
            ImGui::DragFloat("Depth Shift",       &trans.depthShift.x, .01f, -3.f, 3.f);
            ImGui::DragFloat("Depth Scalar",      &trans.depthShift.y, .01f, -90.f,0.f);
            
            ImGui::DragFloatRange2("Depth Range", &trans.depthRange.x, &trans.depthRange.y);
            ImGui::Image(frm, app::toPixels(ivec2(340,191)));
            
        }
        cidx ++;
    }
    ImGui::Text("Depth Frames");
    for(auto& frm : mDepthTex)
        if(frm){
            ImGui::Dummy(app::toPixels(ivec2(0,191)));
            ImGui::Image(frm,   app::toPixels(ivec2(340,-191)));
        }
#endif
}

const nlohmann::json MotionProcessor::toJson(){
    nlohmann::json file, cameras;
    file["distthres"]        = mDistThreshold;
    file["worldthres"]       = mWorldThreshold;
    file["maxskeleton"]      = mMaxSkeleton;
    file["processNoise"]     = mProcessNoise;
    file["measurementNoise"] = mMeasurementNoise;
    file["flatrange"]        = mFlatRange;
    file["fadeduration"]     = {mFadeDuration.x, mFadeDuration.y};
    file["skeletonscale"]    = mSkeletonScale;
    
    cameras = nlohmann::json::array();
    for(auto& c : mCamTransforms){
        auto item = nlohmann::json::object();
        item["name"]        = c.name;
        item["resolution"]  = { c.frame->getWidth(), c.frame->getHeight() };
        item["translate"]   = {c.translate.x, c.translate.y, c.translate.z};
        item["euler"]       = {c.euler.x, c.euler.y, c.euler.z};
        item["depthshift"]  = {c.depthShift.x, c.depthShift.y};
        item["fov"]         = c.fov;
        item["cutoff"]      = c.cutoff;
        item["flip"]        = c.flip;
        item["limit"]       = c.numLimit;
        cameras.push_back(item);
    }
    file["cameras"] = cameras;
    
    return file;
}

void MotionProcessor::load(const nlohmann::json& file){
    mDistThreshold  = file["distthres"];
    mWorldThreshold = file["worldthres"];
    mSkeletonScale  = file["skeletonscale"];
    
    mProcessNoise     = file["processNoise"];
    mMeasurementNoise = file["measurementNoise"];
    for(auto& body : mBodyData)
        body.setNoise(mMeasurementNoise, mProcessNoise);

    mFlatRange        = file["flatrange"];
    mFadeDuration     = vec2(file["fadeduration"][0], file["fadeduration"][1]);

    mMaxSkeleton      = file["maxskeleton"];
    resizeBodyCount();
}
