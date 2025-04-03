#pragma once
#include "Calibrater.h"

namespace SjUtil {
class Unproj {
protected:
    // internal - replicate opencv undistort source
    // since the mobile OpenCV2 build we use do not contain it
    static void _undistort(const CvMat* _src, CvMat* _dst, const CvMat* _cameraMatrix,
                          const CvMat* _distCoeffs,
                          const CvMat* matR, const CvMat* matP) {
        
        double A[3][3], RR[3][3], k[8]={0,0,0,0,0,0,0,0}, fx, fy, ifx, ify, cx, cy;
        CvMat matA  =cvMat(3, 3, CV_64F, A), _Dk;
        CvMat _RR   =cvMat(3, 3, CV_64F, RR);
        const CvPoint2D32f* srcf;
        const CvPoint2D64f* srcd;
        CvPoint2D32f* dstf;
        CvPoint2D64f* dstd;
        int stype, dtype;
        int sstep, dstep;
        int i, j, n, iters = 1;

        cvConvert( _cameraMatrix, &matA );

        if( _distCoeffs ) {
            _Dk = cvMat( _distCoeffs->rows, _distCoeffs->cols,
                CV_MAKETYPE(CV_64F,CV_MAT_CN(_distCoeffs->type)), k);
            cvConvert( _distCoeffs, &_Dk );
            iters = 5;
        }

        if( matR ) {
            cvConvert( matR, &_RR );
        } else
            cvSetIdentity(&_RR);

        if( matP ) {
            double PP[3][3];
            CvMat _P3x3, _PP=cvMat(3, 3, CV_64F, PP);
            cvConvert( cvGetCols(matP, &_P3x3, 0, 3), &_PP );
            cvMatMul( &_PP, &_RR, &_RR );
        }

        srcf = (const CvPoint2D32f*)_src->data.ptr;
        srcd = (const CvPoint2D64f*)_src->data.ptr;
        dstf = (CvPoint2D32f*)_dst->data.ptr;
        dstd = (CvPoint2D64f*)_dst->data.ptr;
        stype = CV_MAT_TYPE(_src->type);
        dtype = CV_MAT_TYPE(_dst->type);
        sstep = _src->rows == 1 ? 1 : _src->step/CV_ELEM_SIZE(stype);
        dstep = _dst->rows == 1 ? 1 : _dst->step/CV_ELEM_SIZE(dtype);

        n = _src->rows + _src->cols - 1;

        fx = A[0][0];
        fy = A[1][1];
        ifx = 1./fx;
        ify = 1./fy;
        cx = A[0][2];
        cy = A[1][2];

        for( i = 0; i < n; i++ ) {
            double x, y, x0, y0;
            if( stype == CV_32FC2 ) {
                x = srcf[i*sstep].x;
                y = srcf[i*sstep].y;
            } else {
                x = srcd[i*sstep].x;
                y = srcd[i*sstep].y;
            }

            x0 = x = (x - cx)*ifx;
            y0 = y = (y - cy)*ify;

            // compensate distortion iteratively
            for( j = 0; j < iters; j++ ) {
                double r2 = x*x + y*y;
                double icdist = (1 + ((k[7]*r2 + k[6])*r2 + k[5])*r2)/(1 + ((k[4]*r2 + k[1])*r2 + k[0])*r2);
                double deltaX = 2*k[2]*x*y + k[3]*(r2 + 2*x*x);
                double deltaY = k[2]*(r2 + 2*y*y) + 2*k[3]*x*y;
                x = (x0 - deltaX)*icdist;
                y = (y0 - deltaY)*icdist;
            }

            double xx = RR[0][0]*x + RR[0][1]*y + RR[0][2];
            double yy = RR[1][0]*x + RR[1][1]*y + RR[1][2];
            double ww = 1./(RR[2][0]*x + RR[2][1]*y + RR[2][2]);
            x = xx*ww;
            y = yy*ww;

            if( dtype == CV_32FC2 ) {
                dstf[i*dstep].x = (float)x;
                dstf[i*dstep].y = (float)y;
            } else {
                dstd[i*dstep].x = x;
                dstd[i*dstep].y = y;
            }
        }
    }
    
    static void undistortPoints( const std::vector<cv::Point2f>& src, std::vector<cv::Point2f>& dst,
                                 const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs ) {
        dst.resize(src.size());
        CvMat _src = cv::Mat(src), _dst = cv::Mat(dst), _cameraMatrix = cameraMatrix;
        CvMat matR, matP, _distCoeffs, *pR=0, *pP=0, *pD=0;
        if( distCoeffs.data )
            pD = &(_distCoeffs = distCoeffs);
        _undistort(&_src, &_dst, &_cameraMatrix, pD, pR, pP);
    }
    
public:
    
    //! Unproject a single point in view space to 3d space
    static ci::vec3 Unproject( const ci::vec3& viewpos,
                               const Calibrater::CameraData& camData ) {
        
        ci::vec3 res;

        std::vector<cv::Point2f> points_undistorted;
        undistortPoints(std::vector<cv::Point2f>{cv::Point2f(viewpos.x, viewpos.y)},
                            points_undistorted, camData.cameraMatrix, camData.distCoeffs);
        
        res.x = 1.f * points_undistorted[0].x * viewpos.z;
        res.y = 1.f * points_undistorted[0].y * viewpos.z;

        res.z = viewpos.z;
        return res;
    }
    
    //! Unproject a vector of points in view space to 3d space
    static std::vector<cv::Point3f> Unproject(
        const std::vector<cv::Point2f>& points,
        const std::vector<float>& Z,
        const Calibrater::CameraData& camData) {
        
        std::vector<cv::Point2f> points_undistorted;
        if (!points.empty()) {
            cv::undistortPoints(points, points_undistorted, 
                camData.cameraMatrix, camData.distCoeffs, 
                cv::noArray(), cv::noArray());
        }

        std::vector<cv::Point3f> result(points_undistorted.size());
        for (size_t idx = 0; idx < points_undistorted.size(); ++idx) {
            const double z = Z.size() == 1 ? Z[0] : Z[idx];
            result[idx] = 
                cv::Point3f(points_undistorted[idx].x * z,
                            points_undistorted[idx].y * z, z);
        }
        return result;
    }

};
}
