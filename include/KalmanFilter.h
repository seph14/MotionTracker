//
//  KalmanFilter.h
//  Tracker
//
//  Created by SEPH LI on 09/02/2024.
//

#ifndef KalmanFilter_h
#define KalmanFilter_h

#include "cinder/CinderMath.h"
#include "cinder/gl/gl.h"

//! Customized kalman filter that only filter depth (z) values
template<int D>
class KalmanFilter {
public:
    typedef typename ci::VECDIM<D, float>::TYPE vecP;
    
protected:
    float x;  // estimated status
    float P;  // estimated covariance
    float Q;  // process noise
    float R;  // measurement noise
    
    // if response is slow -> reduce Q or increase R
    // if output bounces too much -> increase Q or reduce R
    // if initial error is large -> increase P
    vecP  mData;
    
    const void predict(const vecP& u, float dt);

public:
    //! init a kalman filter with default parameters suit for realtime measures
    KalmanFilter(float initial_est_error = 1.0f,
                 float process_noise     = 0.01f,
                 float measurement_noise = 0.1f);
    
    
    //! update with newly measured value and return estimated value
    const vecP& update(const vecP& pos, float dt);
    
    //! reset estimated status and covariance
    void  reset(const vecP& pos);
    
    //! force the filter to reset with next measurement value
    void  clear();

    //! check if we have valid estimated value
    const bool valid ();
    
    const vecP& pos() const { return mData; }
    
    const void setMeasurementNoise(float noise) {
        Q = noise;
    }
    
    const void setProcessNoise    (float noise) {
        R = noise;
    }
};


typedef KalmanFilter<4> Kalman4f;
typedef KalmanFilter<3> Kalman3f;
typedef KalmanFilter<2> Kalman2f;

#endif /* KalmanFilter_h */
