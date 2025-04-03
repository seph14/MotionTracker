//
//  MotionProcessor.h
//  Tracker
//
//  Created by SEPH LI on 07/02/2024.
//

#ifndef MotionProcessor_h
#define MotionProcessor_h

#include "cinder/Capture.h"
#include "BodyData.h"
#include "json.hpp"
#include "cinder/CameraUi.h"
#include "cinder/gl/TextureFont.h"
#include "Resources.h"
#include "Calibrater.h"
#include "Unproj.h"

class MotionProcessor;
typedef std::shared_ptr<MotionProcessor> MotionProcessorRef;

template<class T, class Allocator = std::allocator<T>>
class ThreadSafeDeque {
public:
    void pop_front_waiting(T &t) {
        // unique_lock can be unlocked, lock_guard can not
        std::unique_lock<std::mutex> lock{ mutex }; // locks
        while(deque.empty())
            condition.wait(lock); // unlocks, sleeps and relocks when woken up
        t = deque.front();
        deque.pop_front();
    } // unlocks as goes out of scope

    void push_back(const T &t) {
        {
            std::lock_guard<std::mutex> lock{ mutex };
            deque.push_back(t);
        }
        condition.notify_one(); // wakes up pop_front_waiting
    }
    
    const size_t size() { return deque.size(); }
    
private:
    std::deque<T, Allocator>    deque;
    std::mutex                  mutex;
    std::condition_variable condition;
};

class MotionProcessor {
public:
    struct SkeletonData {
        ml::BodyData data;
        bool         updated;
        int32_t      idx;
        float        fadeIn, fadeOut; // controls fade in-out
        uint8_t      camIdx;
        int32_t      overlapRef;
        
        // final world positions
        std::array<ci::vec3, 19> wpos;
    };
    
    enum class RenderMode {
        ThreeD = 0, Depth = 1, RGB = 2
    };
    
    struct CamTransform {
        std::string      name;
        ci::vec3         translate;
        ci::vec3         euler;
        float            fov;
        float            aspectRatio;
        ci::CameraPersp  cam;
        ci::CaptureRef   capture;
        ci::Surface8uRef frame;
        
        // camera transform
        SjUtil::Calibrater::CameraData data;
        Transformation   transform;
        ci::mat4         worldMatrix;
        
        ci::ivec4        viewRange;
        bool             updated, flip;
        // cut off ratio to avoid duplicate figures
        float            cutoff;
        // limitation on human tracked
        uint32_t         numLimit;
        // range reported from depth map
        ci::vec2         depthRange;
        // camera based depth scalar
        ci::vec2         depthShift;
        
        CamTransform() : cutoff(.0f), depthShift(26.f,-60.f),
            //depthThreshold(.05f),
            fov(90.f),aspectRatio(-1.f), numLimit(8), updated(false), flip(false) {}
    };
    
protected:
    // max no. of people tracked
    int      mMaxSkeleton;
    // distance threshold
    float    mDistThreshold;
    // world dstance threshold for camera overlap
    float    mWorldThreshold;
    // kalman values
    float    mMeasurementNoise, mProcessNoise;
    // 2D range
    float    mFlatRange;
    // fade duration
    ci::vec2 mFadeDuration;
    // unprojected skeleton scale
    float    mSkeletonScale;
    bool     mNoRender;
    
    // drawing camera/debug info
    ci::gl::TextureFontRef    mTextureFont;
    
    // check if skeleton is properly tracked
    const bool isSkeletonStable(const SkeletonData& data);
    
    RenderMode              mMode;
    // ml & interaction
    bool                    mDrawSkeleton, mDrawAllSkeleton, mAllFeedProcessed,
                            mShouldQuit, mPrevTracked, mTracked, mCamInited;
    std::atomic_uint32_t    mMLFrms;
    uint32_t                mFeedFrms, mProcessedFeedNum;
    float                   mUpdateTime, mMLFps, mFeedFps, mLastSampleTime;
    std::atomic_bool        mProcessRgb;
    
    std::vector<CamTransform> mCamTransforms;
    std::vector<ci::gl::Texture2dRef>
                            mCapTex, mDepthTex;
    
    // put rgb and depth frames bundles for every 2 cameras
    std::vector<ci::Surface8uRef>   mDepthInput;
    std::vector<ci::Channel32fRef>  mDepthFrame;
    
    // pipe-in camera feed to ML process as soon as a new frame is available
    struct CamFrame {
        uint8_t          idx;
        uint32_t         limit;
        ci::Surface8uRef frame;
        ci::Surface8uRef depth;
    };
    ThreadSafeDeque<CamFrame>     mQueIn;
    
    struct MLFrame {
        uint8_t                   idx;
        ci::Channel32fRef         depth;
        ci::vec2                  depthRange;
        std::vector<ml::BodyData> bodyData;
    };
    ThreadSafeDeque<MLFrame>      mQueOut;
    
    // depth calibration helper
    std::vector<float>           mRawDepth, mRealDepth;
    std::vector<ci::vec2>        mScrDepthPnts;
    int8_t                       mDepthTargetIdx;
    
    ci::CameraPersp              mCam;
    ci::CameraUi                 mCamUi;
    int                          mActiveNum, mDetectedNum;
    ci::ivec2                    mRgbBound;
    std::vector<std::pair<std::string, ci::Capture::DeviceIdentifier>> mCamNames;
    ci::Surface8uRef             mDataFrame;
    std::shared_ptr<std::thread> mProcessThd;
    std::mutex                   mtxBody;
    
    std::vector<SkeletonData> mTargetBody;
    std::vector<ml::BodyData> mBodyData, mTmpData;
    std::vector<ci::Color>    mBodyColor;
    std::vector<int32_t>      mRemovedIdx;
    
    MotionProcessor();
    MotionProcessor(const nlohmann::json& file);
    
    void initParams();
    void loadShaders();
    void findCameras();
    void setupCamera(CamTransform& camTrans, uint8_t idx);
    void mapBody( SkeletonData& body, const CamTransform& camTrans );
    const ci::vec3 unproject(const ci::vec3& p, const CamTransform& cam);
    
    ci::Surface8uRef resizeImage(ci::Surface8uRef src,
                                 const ci::ivec2& targetSize);
    void processBodyTracking();
    void resizeBodyCount();
    void resetBodyIdx();
    void remapBodyIdx(const int& oldIdx, const int& newIdx);
    void drawSkeleton3d(const std::array<ci::vec3,19>& wpos);
    void drawDepthUi();
    
    //! use linear regression to map relative depth value to metric values
    ci::vec2 calculateDepthShift(const std::vector<float>& rawDepth,
                                 const std::vector<float>& targetDepth);
    
public:
    ~MotionProcessor();
    
    //! init a default motion processor that init with facetime camera
    static MotionProcessorRef create();
    
    //! init with json file
    static MotionProcessorRef create(const nlohmann::json& file);
    
    bool  update(const float& dt, const float& time);
    
    void  printAvailableCameraNames();
    
    //! num of detected skeletons
    const size_t availableSkeletonCnt();
    //! get the skeleton data, first vec3 is metadata of skeleton index, detected ratio, stability, then, followed by  18 standard joint world positions
    std::vector<ci::vec3> skeletonData(const size_t& idx);
    
    void draw  (const ci::Rectf& bound);
    void drawUi();

    const size_t numSections()   { return mCamTransforms.size(); }
    const ci::ivec2& rgbBounds() { return mRgbBound; }
    void  drawRGBComposition();
    
    const bool draw3d() { return mMode == RenderMode::ThreeD; }
    
    //! add a test point on the depth feed, to calculate linear regression parameters to the depth feed
    void addDepthPoint(const ci::vec2& pnt);
    
    const ci::Surface8uRef dataFrame()  { return mDataFrame; }
    const ci::Surface8uRef rgbFrame ()  { return mCamTransforms[0].frame; }
    const std::vector<SkeletonData>& skeletonData() { return mTargetBody; }
    
    const ci::gl::Texture2dRef rgbTex(const uint32_t& idx)   { return mCapTex[idx]; }
    const ci::gl::Texture2dRef depthTex(const uint32_t& idx) { return mDepthTex[idx]; }
    
    //! save and load functions
    const nlohmann::json toJson();
    void  load(const nlohmann::json& file);
};

#endif /* MotionProcessor_h */
