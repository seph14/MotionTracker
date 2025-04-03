//
//  PoseHandler.h
//  Tracker
//
//  Created by SEPH LI on 07/02/2024.
//

#ifndef PoseHandler_h
#define PoseHandler_h

#include "stdlib.h"
#include <vector>
#include "BodyData.h"

namespace ml {
    class PoseTracker {  
    public:
        
        static void init();
        static void release();
        
        //! convert surface to cgimageref for later processing
        static CGImageRef convetSurface(ci::Surface8uRef surf);
        
        //! run a vison process to process all skeleton data
        static std::vector<BodyData>& process( CGImageRef surf );
    };
};

#endif /* PoseHandler_h */
