//
//  DepthHandler.h
//  YourStory
//
//  Created by Seph Li on 15/07/2021.
//

#ifndef DepthHandler_h
#define DepthHandler_h
#include "Resources.h"
#include "cinder/gl/gl.h"

namespace ml{
    class DepthHandler {
    public:
        static void init();
        static void release();
        
        const static ci::ivec2 getExpectedSize();
        const static ci::ivec2 getOutputSize();
        const static ci::vec2  getDepthRange();
        
        //! run DepthAnythingV2 pass to get a single channel depth feed
        static ci::Channel32fRef process( ci::Surface8u& surf );
    };
};

#endif /* DepthHandler_h */
