//
//  Calibrater.cpp
//  OpenCVTest
//
//  Created by SEPH LI on 05/03/2025.
//

#include "Calibrater.h"
#include "cinder/Log.h"
#include "cinder/Utilities.h"
#include <thread>

using namespace SjUtil;
using namespace ci;
using namespace std;

#ifdef CINDER_MSW
bool Calibrater::mPaused         = false;
bool Calibrater::mSingleProcess  = false;
bool Calibrater::mStereoProcess  = false;
ci::Surface8uRef Calibrater::mMainSurf = nullptr;
ci::Surface8uRef Calibrater::mSubSurf  = nullptr;
#endif

const nlohmann::json Calibrater::CameraData::toJson() {
    nlohmann::json file, coeff;
    file["intrinsic"] = { 
        cameraMatrix.at<double>(0,0),cameraMatrix.at<double>(1,0), cameraMatrix.at<double>(2,0),
        cameraMatrix.at<double>(0,1),cameraMatrix.at<double>(1,1), cameraMatrix.at<double>(2,1),
        cameraMatrix.at<double>(0,2),cameraMatrix.at<double>(1,2), cameraMatrix.at<double>(2,2) };

    for (int n = 0; n < distCoeffs.rows; n++)
        coeff.push_back(distCoeffs.at<double>(n, 0));
    file["coeff"] = coeff;

    return file;
}

void Calibrater::CameraData::load(const nlohmann::json& file) {
    auto& mat   = file["intrinsic"];
    cameraMatrix.at<double>(0, 0) = mat[0]; cameraMatrix.at<double>(1, 0) = mat[1]; cameraMatrix.at<double>(2, 0) = mat[2];
    cameraMatrix.at<double>(0, 1) = mat[3]; cameraMatrix.at<double>(1, 1) = mat[4]; cameraMatrix.at<double>(2, 1) = mat[5];
    cameraMatrix.at<double>(0, 2) = mat[6]; cameraMatrix.at<double>(1, 2) = mat[7]; cameraMatrix.at<double>(2, 2) = mat[8];
    
    auto& coeff = file["coeff"];
    auto num = static_cast<uint32_t>(coeff.size());
    distCoeffs = cv::Mat::zeros(num, 1, CV_64F);
    for(uint32_t i = 0; i < num; i++)
        distCoeffs.at<double>(i, 0) = (double)coeff[i] / 10e3;
}

void Calibrater::CameraData::generateDummy(){
    cameraMatrix.at<double>(0, 0) = 10310.96; cameraMatrix.at<double>(1, 0) = 0.; cameraMatrix.at<double>(2, 0) = 0.;
    cameraMatrix.at<double>(0, 1) = 0.;       cameraMatrix.at<double>(1, 1) = 5799.92; cameraMatrix.at<double>(2, 1) = 0.;
    cameraMatrix.at<double>(0, 2) = 640;      cameraMatrix.at<double>(1, 2) = 360; cameraMatrix.at<double>(2, 2) = 1;
    
    distCoeffs = cv::Mat::zeros(5, 1, CV_64F);
    distCoeffs.at<double>(0,0) = 12.f       / 10e3;
    distCoeffs.at<double>(1,0) = -1400.f    / 10e3;
    distCoeffs.at<double>(2,0) = -0.5423f   / 10e3;
    distCoeffs.at<double>(3,0) = 0.0198f    / 10e3;
    distCoeffs.at<double>(4,0) = -9.25f     / 10e3;
}

#ifdef CINDER_MSW
void Calibrater::resumeProcess() {
    mPaused = false;
}

ci::Surface8uRef Calibrater::getColorImage0() {
    return mMainSurf;
}

ci::Surface8uRef Calibrater::getColorImage1() {
    return mSubSurf;
}

double Calibrater::calibrateSingle(
    std::vector<std::vector<cv::Point2f>>& corners_list,
    const ChessboardPattern& board,
    float aspectRatio, const cv::Size& imageSize, CameraData& data) {

    // prepare 3d object points
    vector<vector<cv::Point3f>> objectPoints(1);
    float grid_width = board.square_length * (board.pattern.width - 1);
    for (int i = 0; i < board.pattern.height; ++i)
        for (int j = 0; j < board.pattern.width; ++j)
            objectPoints[0].emplace_back(cv::Point3f(j * board.square_length, i * board.square_length, 0));
    objectPoints[0][board.pattern.width - 1].x = objectPoints[0][0].x + grid_width;
    auto newObjPnts = objectPoints[0];
    objectPoints.resize(corners_list.size(), objectPoints[0]);

    // init matrix
    data.cameraMatrix.at<double>(0, 0) = aspectRatio;
    
    //Find intrinsic and extrinsic camera parameters
    vector<cv::Mat> rvecs, tvecs;
    double rms = cv::calibrateCamera(objectPoints, corners_list, imageSize,
        data.cameraMatrix, data.distCoeffs, rvecs, tvecs,
        cv::CALIB_FIX_ASPECT_RATIO);

    string val = "New board corners: \r\n"
        + toString(newObjPnts[0]) + "\r\n"
        + toString(newObjPnts[board.pattern.width - 1]) + "\r\n"
        + toString(newObjPnts[board.pattern.width * (board.pattern.height - 1)]) + "\r\n"
        + toString(newObjPnts.back());
    CI_LOG_I(val);
    CI_LOG_I("Re-projection error reported by calibrateCamera: " << rms);

    bool ok = cv::checkRange(data.cameraMatrix) && cv::checkRange(data.distCoeffs);
    if (!ok) return -1.;

    objectPoints.clear();
    objectPoints.resize(corners_list.size(), newObjPnts);

    vector<cv::Point2f> imagePoints2;
    size_t totalPoints = 0;
    double totalErr = 0, err;

    for (size_t i = 0; i < objectPoints.size(); ++i) {
        cv::projectPoints(objectPoints[i], rvecs[i], tvecs[i],
            data.cameraMatrix, data.distCoeffs, imagePoints2);
        err = cv::norm(corners_list[i], imagePoints2, cv::NORM_L2);
        size_t n = objectPoints[i].size();
        totalErr += err * err;
        totalPoints += n;
    }

    return std::sqrt(totalErr / totalPoints);
}

Calibrater::CameraData Calibrater::calibrateSingleCamera( ci::CaptureRef capture, bool& success,
                                              const ChessboardPattern& board, const uint8_t calibration_frame ) {
    if(!capture->isCapturing()) capture->start();
    
    mPaused = false;
    cv::Mat     color_image, gray_image;
    cv::Size    imageSize;
    CameraData  data;
    Mode        mode = Mode::SINGLE_DETECTION;
    vector<vector<cv::Point2f>> corners_list;
    
    auto processCorners = [&color_image, &board](vector<cv::Point2f>& pointBuf) {
        int chessBoardFlags = cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FAST_CHECK;
        return cv::findChessboardCorners(color_image, board.pattern, pointBuf , chessBoardFlags);
    };
    
    while(!mPaused) {
        if (!capture->checkNewFrame()) {
            std::this_thread::sleep_for(30ms);
            continue;
        }

        color_image = toOcvRef(*capture->getSurface());
        imageSize   = color_image.size();
        
        //if( mode == Mode::SINGLE_DETECTION &&  )
        vector<cv::Point2f> curr_corners;
        if(processCorners(curr_corners)) {
            // improve the found corners' coordinate accuracy for chessboard
            cvtColor(color_image, gray_image, cv::COLOR_BGR2GRAY);
            cv::cornerSubPix( gray_image, curr_corners,
                              cv::Size(11,11), cv::Size(-1,-1),
                              cv::TermCriteria( cv::TermCriteria::EPS+cv::TermCriteria::COUNT, 30, 0.0001 ));
            
            // upload to output surf so we can draw
            cv::drawChessboardCorners(color_image, board.pattern, curr_corners, true);
            mMainSurf = Surface8u::create(fromOcv(color_image));
            CI_LOG_I( "corners found: " << int(corners_list.size()) << "/" << (int)calibration_frame );
      
            corners_list.emplace_back(curr_corners);
            std::this_thread::sleep_for(500ms);
        } else {
            mMainSurf = capture->getSurface();
            std::this_thread::sleep_for(30ms);
        }
        
        if (corners_list.size() >= calibration_frame) {
            auto err = calibrateSingle(corners_list, board, capture->getAspectRatio(), imageSize, data);
            CI_LOG_I("Average err: " << err);
            break;
        }
    }
    
    return data;
}

/*
cv::Matx33f Calibrater::calibration_to_color_camera_matrix(const k4a::calibration &cal) {
    const k4a_calibration_intrinsic_parameters_t::_param &i = cal.color_camera_calibration.intrinsics.parameters.param;
    cv::Matx33f camera_matrix = cv::Matx33f::eye();
    camera_matrix(0, 0) = i.fx;
    camera_matrix(1, 1) = i.fy;
    camera_matrix(0, 2) = i.cx;
    camera_matrix(1, 2) = i.cy;
    return camera_matrix;
}

vector<float> Calibrater::calibration_to_color_camera_dist_coeffs(const k4a::calibration &cal) {
    const k4a_calibration_intrinsic_parameters_t::_param &i = cal.color_camera_calibration.intrinsics.parameters.param;
    return { i.k1, i.k2, i.p1, i.p2, i.k3, i.k4, i.k5, i.k6 };
}
*/
bool Calibrater::find_chessboard_corners_helper(
    const cv::Mat &main_color_image, const cv::Mat &secondary_color_image,
    const cv::Size &chessboard_pattern,
    vector<cv::Point2f> &main_chessboard_corners, vector<cv::Point2f> &secondary_chessboard_corners) {
    
    int chessBoardFlags = cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FAST_CHECK;
    bool found_chessboard_main      = cv::findChessboardCorners(
        main_color_image,      chessboard_pattern, main_chessboard_corners, chessBoardFlags);
    bool found_chessboard_secondary = cv::findChessboardCorners(
        secondary_color_image, chessboard_pattern, secondary_chessboard_corners, chessBoardFlags);

    // Cover the failure cases where chessboards were not found in one or both images.
    if (!found_chessboard_main || !found_chessboard_secondary) {
        if (found_chessboard_main) 
            CI_LOG_I( "Could not find the chessboard corners in the secondary image. Trying again...");
        // Likewise, if the chessboard was found in the secondary image, it was not found in the main image.
        else if (found_chessboard_secondary)
            CI_LOG_I( "Could not find the chessboard corners in the main image. Trying again...");
        // The only remaining case is the corners were in neither image.
        else 
            CI_LOG_I( "Could not find the chessboard corners in either image. Trying again...");
        return false;
    }

    cv::Mat gray_image;
    
    cvtColor(main_color_image, gray_image, cv::COLOR_BGR2GRAY);
    cv::cornerSubPix(gray_image, main_chessboard_corners,
        cv::Size(11, 11), cv::Size(-1, -1),
        cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.0001));
    
    cvtColor(secondary_color_image, gray_image, cv::COLOR_BGR2GRAY);
    cv::cornerSubPix(gray_image, secondary_chessboard_corners,
        cv::Size(11, 11), cv::Size(-1, -1),
        cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.0001));

    // Before we go on, there's a quick problem with calibration to address.  Because the chessboard looks the same when
    // rotated 180 degrees, it is possible that the chessboard corner finder may find the correct points, but in the
    // wrong order.

    // A visual:
    //        Image 1                  Image 2
    // .....................    .....................
    // .....................    .....................
    // .........xxxxx2......    .....xxxxx1..........
    // .........xxxxxx......    .....xxxxxx..........
    // .........xxxxxx......    .....xxxxxx..........
    // .........1xxxxx......    .....2xxxxx..........
    // .....................    .....................
    // .....................    .....................

    // The problem occurs when this case happens: the find_chessboard() function correctly identifies the points on the
    // chessboard (shown as 'x's) but the order of those points differs between images taken by the two cameras.
    // Specifically, the first point in the list of points found for the first image (1) is the *last* point in the list
    // of points found for the second image (2), though they correspond to the same physical point on the chessboard.

    // To avoid this problem, we can make the assumption that both of the cameras will be oriented in a similar manner
    // (e.g. turning one of the cameras upside down will break this assumption) and enforce that the vector between the
    // first and last points found in pixel space (which will be at opposite ends of the chessboard) are pointing the
    // same direction- so, the dot product of the two vectors is positive.

    cv::Vec2f main_image_corners_vec        = main_chessboard_corners.back() - main_chessboard_corners.front();
    cv::Vec2f secondary_image_corners_vec   = secondary_chessboard_corners.back() - secondary_chessboard_corners.front();
    if (main_image_corners_vec.dot(secondary_image_corners_vec) <= 0.f) 
        std::reverse(secondary_chessboard_corners.begin(), secondary_chessboard_corners.end());
    
    return true;
}

Transformation Calibrater::stereo_calibration(
    CameraData &main_calib,
    CameraData &secondary_calib,
    const vector<vector<cv::Point2f>> &main_chessboard_corners_list,
    const vector<vector<cv::Point2f>> &secondary_chessboard_corners_list,
    const cv::Size &image_size,
    const ChessboardPattern& board) {
    // We have points in each image that correspond to the corners that the findChessboardCorners function found.
    // However, we still need the points in 3 dimensions that these points correspond to. Because we are ultimately only
    // interested in find a transformation between two cameras, these points don't have to correspond to an external
    // "origin" point. The only important thing is that the relative distances between points are accurate. As a result,
    // we can simply make the first corresponding point (0, 0) and construct the remaining points based on that one. The
    // order of points inserted into the vector here matches the ordering of findChessboardCorners. The units of these
    // points are in millimeters, mostly because the depth provided by the depth cameras is also provided in
    // millimeters, which makes for easy comparison.
    vector<cv::Point3f> chessboard_corners_world;
    for (int h = 0; h < board.pattern.height; ++h) {
        for (int w = 0; w < board.pattern.width; ++w) {
            chessboard_corners_world.emplace_back(
                cv::Point3f{ w * board.square_length, h * board.square_length, 0.f });
        }
    }

    // Calibrating the cameras requires a lot of data. OpenCV's stereoCalibrate function requires:
    // - a list of points in real 3d space that will be used to calibrate*
    // - a corresponding list of pixel coordinates as seen by the first camera*
    // - a corresponding list of pixel coordinates as seen by the second camera*
    // - the camera matrix of the first camera
    // - the distortion coefficients of the first camera
    // - the camera matrix of the second camera
    // - the distortion coefficients of the second camera
    // - the size (in pixels) of the images
    // - R: stereoCalibrate stores the rotation matrix from the first camera to the second here
    // - t: stereoCalibrate stores the translation vector from the first camera to the second here
    // - E: stereoCalibrate stores the essential matrix here (we don't use this)
    // - F: stereoCalibrate stores the fundamental matrix here (we don't use this)
    //
    // * note: OpenCV's stereoCalibrate actually requires as input an array of arrays of points for these arguments,
    // allowing a caller to provide multiple frames from the same camera with corresponding points. For example, if
    // extremely high precision was required, many images could be taken with each camera, and findChessboardCorners
    // applied to each of those images, and OpenCV can jointly solve for all of the pairs of corresponding images.
    // However, to keep things simple, we use only one image from each device to calibrate.  This is also why each of
    // the vectors of corners is placed into another vector.
    //
    // A function in OpenCV's calibration function also requires that these points be F32 types, so we use those.
    // However, OpenCV still provides doubles as output, strangely enough.
    vector<vector<cv::Point3f>> chessboard_corners_world_nested_for_cv(
        main_chessboard_corners_list.size(),
        chessboard_corners_world);

    // Finally, we'll actually calibrate the cameras.
    // Pass secondary first, then main, because we want a transform from secondary to main.
    Transformation tr;
    
    double error = cv::stereoCalibrate(chessboard_corners_world_nested_for_cv,
        secondary_chessboard_corners_list,
        main_chessboard_corners_list,
        secondary_calib.cameraMatrix, secondary_calib.distCoeffs,
        main_calib.cameraMatrix, main_calib.distCoeffs,
        image_size,
        tr.R, // output
        tr.t, // output
        cv::noArray(),
        cv::noArray(),
        //cv::CALIB_FIX_ASPECT_RATIO);
        //CV_CALIB_SAME_FOCAL_LENGTH | CV_CALIB_ZERO_TANGENT_DIST);
        cv::CALIB_FIX_INTRINSIC | cv::CALIB_RATIONAL_MODEL | cv::CALIB_CB_FAST_CHECK);
    CI_LOG_I( "Finished calibrating, got error of " << error);
    return tr;
}

std::vector<Transformation> Calibrater::calibrateStereoCameras(
    std::vector<ci::CaptureRef> devices, 
    std::vector<CameraData>&    data,
    const ChessboardPattern&    board, 
    uint8_t                     calibration_frame) {

    // ref to captures, and update itermediate camera data
    auto master_device      = devices[0];
    data.resize(devices.size());
    CameraData& master_data = data[0];
    
    mPaused = false;
    std::vector<Transformation> result;
    
    for (int i = 1; i < devices.size(); i++) {
        
        auto        sub_device  = devices[i];
        CameraData& sub_data    = data[i];

        vector<vector<cv::Point2f>> main_chessboard_corners_list;
        vector<vector<cv::Point2f>> secondary_chessboard_corners_list;
        
        while (!mPaused) {
            // not doing sync across cameras for now
            if (!master_device->checkNewFrame() || !sub_device->checkNewFrame()) {
                std::this_thread::sleep_for(30ms);
                continue;
            }

            // fetch both feed
            cv::Mat cv_main_color_image         = toOcvRef(*master_device->getSurface());
            cv::Mat cv_secondary_color_image    = toOcvRef(*sub_device->getSurface());
            
            vector<cv::Point2f> main_chessboard_corners;
            vector<cv::Point2f> secondary_chessboard_corners;
            bool got_corners = find_chessboard_corners_helper(
                cv_main_color_image,     cv_secondary_color_image, board.pattern,
                main_chessboard_corners, secondary_chessboard_corners
            );
            
            if (got_corners) {
                main_chessboard_corners_list.emplace_back     (main_chessboard_corners);
                secondary_chessboard_corners_list.emplace_back(secondary_chessboard_corners);
                cv::drawChessboardCorners(cv_main_color_image,      board.pattern, main_chessboard_corners,      true);
                cv::drawChessboardCorners(cv_secondary_color_image, board.pattern, secondary_chessboard_corners, true);
            
                mMainSurf = Surface8u::create(fromOcv(cv_main_color_image));
                mSubSurf  = Surface8u::create(fromOcv(cv_secondary_color_image));
                CI_LOG_I( "corners found: " << int(main_chessboard_corners_list.size()) << "/" << (int)calibration_frame );
                std::this_thread::sleep_for(std::chrono::microseconds(500));
            } else {
                mMainSurf = master_device->getSurface();
                mSubSurf  = sub_device->getSurface();
            }
            
            // Get required frames before doing calibration.
            if (main_chessboard_corners_list.size() >= calibration_frame) {
                if (i == 1) {
                    auto err = calibrateSingle(main_chessboard_corners_list, board, 
                        master_device->getAspectRatio(), cv_main_color_image.size(), master_data);
                    CI_LOG_I("Calibrate main camera parameters with err: " << err);
                }

                {
                    auto err = calibrateSingle(secondary_chessboard_corners_list, board,
                        sub_device->getAspectRatio(), cv_secondary_color_image.size(), sub_data);
                    CI_LOG_I("Calibrate sub " << i << " camera parameters with err: " << err);
                }

                CI_LOG_I( "Calculating calibration pair " << (i-1) << "..." );
                result.push_back(stereo_calibration(
                    master_data,
                    sub_data,
                    
                    main_chessboard_corners_list,
                    secondary_chessboard_corners_list,
                    
                    cv_main_color_image.size(), board));

                auto& trans = result.back();
                auto model = trans.to_model(1.f);


                CI_LOG_I( "done -> offset:" << (model * vec4(0.f,0.f,0.f,1.f)) );
                break;
            }
        }
    }

    return result;
}

#endif
