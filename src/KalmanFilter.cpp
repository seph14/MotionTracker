//
//  KalmanFilter.cpp
//  Tracker
//
//  Created by SEPH LI on 09/02/2024.
//

#include "KalmanFilter.h"
#include "cinder/Log.h"
#include "cinder/app/App.h"

using namespace ci;
using namespace std;

template<int D>
KalmanFilter<D>::KalmanFilter(float initial_est_error,
                              float process_noise,
                              float measurement_noise) {
    x = NAN;
    P = initial_est_error;
    Q = process_noise;
    R = measurement_noise;
}

template<int D>
const void KalmanFilter<D>::predict(const KalmanFilter<D>::vecP& u, float dt) {
    P += Q;
}

template<int D>
const typename KalmanFilter<D>::vecP& KalmanFilter<D>::update(
    const typename KalmanFilter<D>::vecP& pos, float dt) {

    if(glm::isnan(x)){
        reset(pos);
    } else {
        predict(pos, dt);
        const float K = P / (P + R);
        x += K * (pos[2] - x);
        P *= (1.f - K);
        mData.x = pos.x;
        mData.y = pos.y;
        mData[2]= x;
    }

    return mData;
}

template<int D>
void KalmanFilter<D>::reset(const typename KalmanFilter<D>::vecP& pos) {
    x     = pos[2];
    mData = pos;
    P     = 1.f;
}

template<int D>
void KalmanFilter<D>::clear() {
    x   = NAN;
}

template<int D>
const bool KalmanFilter<D>::valid() {
    return !glm::isnan(x);
}

template class KalmanFilter<2>;
template class KalmanFilter<3>;
template class KalmanFilter<4>;
