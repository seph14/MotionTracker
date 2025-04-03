//
//  Calibrater.h
//  Motion Tracker
//
//  Created by SEPH LI on 05/03/2025.
//

#ifndef Calibrater_h
#define Calibrater_h

#include "CinderOpenCV.h"
#include "cinder/Capture.h"
#include "Transformation.h"
#include <vector>

namespace SjUtil {
class Calibrater {
public:
    enum class Mode {
        SINGLE_DETECTION, SINGLE_CALIBRATED,
        STEREO_DETECTION, STEREO_CALIBRATED
    };
    
    //! Defines camera data that contains distortion parameters
    struct CameraData {
        cv::Mat cameraMatrix, distCoeffs;
        
        CameraData() {
            cameraMatrix = cv::Mat::eye  (3, 3, CV_64F);
            distCoeffs   = cv::Mat::zeros(8, 1, CV_64F);
        }

        inline ci::vec4 getFC() const {
            return ci::vec4(
                (float)cameraMatrix.at<double>(0, 0), //f_x
                (float)cameraMatrix.at<double>(1, 1), //f_y
                (float)cameraMatrix.at<double>(0, 2), //c_x
                (float)cameraMatrix.at<double>(1, 2)  //c_y
            );
        }
        
        const nlohmann::json toJson();
        void load(const nlohmann::json& file);
        void generateDummy();
    };
    
    //! Defines chessboard pattern for calibration
    struct ChessboardPattern {
        cv::Size pattern;
        float    square_length;
            
        ChessboardPattern() : pattern(9, 6), square_length(26.5f)
        {}
    };

// calibration functions are not used
// for this build, we do not use a full opencv build on mac
#ifdef CINDER_MSW
protected:
    static bool             mPaused, mSingleProcess, mStereoProcess;
    static ci::Surface8uRef mMainSurf, mSubSurf;
    
    static double calibrateSingle( 
        std::vector<std::vector<cv::Point2f>>& corners_list, 
        const ChessboardPattern& board, 
        float aspectRatio, const cv::Size& imageSize, CameraData& data );

    static bool find_chessboard_corners_helper(
        const cv::Mat             &main_color_image,
        const cv::Mat             &secondary_color_image,
        const cv::Size             &chessboard_pattern,
        std::vector<cv::Point2f> &main_chessboard_corners,
        std::vector<cv::Point2f> &secondary_chessboard_corners);
        
    static Transformation stereo_calibration(
        CameraData                            &main_calib,
        CameraData                            &secondary_calib,
        const std::vector<std::vector<cv::Point2f>> &main_chessboard_corners_list,
        const std::vector<std::vector<cv::Point2f>> &secondary_chessboard_corners_list,
        const cv::Size                              &image_size,
        const ChessboardPattern& board);

public:
    static CameraData calibrateSingleCamera( ci::CaptureRef capture, bool& success,
                                            const ChessboardPattern& board = ChessboardPattern(),
                                            const uint8_t calibration_frame = 20);
    
    static std::vector<Transformation> calibrateStereoCameras(
        std::vector<ci::CaptureRef> devices,
        std::vector<CameraData>&    data,
        const ChessboardPattern&    board = ChessboardPattern(),
        uint8_t                     calibration_frame = 20
    );

    static ci::Surface8uRef getColorImage0();
    static ci::Surface8uRef getColorImage1();

    static void resumeProcess();
#endif
};
}

#endif /* Calibrater_h */
