//
//  DepthHandler.cpp
//  YourStory
//
//  Created by Seph Li on 15/07/2021.
//

#include "DepthHandler.h"

#include "cinder/Cinder.h"
#include "cinder/gl/gl.h"
#include "cinder/app/App.h"
#include "cinder/Log.h"

#include "DepthAnythingV2SmallF16.h"

#import  <CoreML/CoreML.h>
#import  <Vision/Vision.h>
#import <Accelerate/Accelerate.h>

using namespace std;
using namespace ci;
using namespace ml;

static const int INPUT_WIDTH  = 518;
static const int INPUT_HEIGHT = 392;
static const int OUTPUT_WIDTH = 518;
static const int OUTPUT_HEIGHT= 392;

API_AVAILABLE(macos(14.0)) DepthAnythingV2SmallF16 *_depthModel;
ci::Channel32fRef   mSurfaceDepth;
ci::vec2            mDepthRange;
NSDictionary        * attributes;
std::atomic_bool    mInited;

const ci::ivec2 DepthHandler::getExpectedSize(){ return ivec2(INPUT_WIDTH, INPUT_HEIGHT); }
const ci::ivec2 DepthHandler::getOutputSize(){ return ivec2(OUTPUT_WIDTH, OUTPUT_HEIGHT); }
const ci::vec2  DepthHandler::getDepthRange() { return mDepthRange; }

void DepthHandler::init(){
    attributes = @{
        (NSString *)kCVPixelBufferIOSurfacePropertiesKey : @{},
        (NSString *)kCVPixelBufferCGImageCompatibilityKey : @(YES),
        (NSString *)kCVPixelBufferCGBitmapContextCompatibilityKey : @(YES),
    };
    
    if (@available(macOS 14.0, *)) {
        _depthModel = [[DepthAnythingV2SmallF16 alloc] init];
    }
    mInited = true;
}

void DepthHandler::release(){
    mInited = false;
    mSurfaceDepth.reset();
    if (@available(macOS 14.0, *)) {
        //[_depthModel release];
    }
}

CVPixelBufferRef ConvertPixelBufferTo32Float(CVPixelBufferRef sourceBuffer) {
    /*
    // Validate input format
    OSType sourceFormat = CVPixelBufferGetPixelFormatType(sourceBuffer);
    if (sourceFormat != kCVPixelFormatType_OneComponent16Half) {
        NSLog(@"Invalid source pixel format");
        return NULL;
    }
     */
    
    // Get source dimensions
    size_t width  = CVPixelBufferGetWidth(sourceBuffer);
    size_t height = CVPixelBufferGetHeight(sourceBuffer);

    // Create destination pixel buffer
    CVPixelBufferRef destinationBuffer = NULL;
    NSDictionary *destinationAttributes = @{
        (id)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_OneComponent32Float),
        (id)kCVPixelBufferWidthKey: @(width),
        (id)kCVPixelBufferHeightKey: @(height),
        (id)kCVPixelBufferMetalCompatibilityKey: @YES // Optional, depending on use case
    };
    
    CVReturn status = CVPixelBufferCreate(
        kCFAllocatorDefault,
        width,
        height,
        kCVPixelFormatType_OneComponent32Float,
        (__bridge CFDictionaryRef)destinationAttributes,
        &destinationBuffer
    );

    if (status != kCVReturnSuccess || !destinationBuffer) {
        NSLog(@"Failed to create destination pixel buffer");
        return NULL;
    }

    // Lock buffers for memory access
    CVPixelBufferLockBaseAddress(sourceBuffer, kCVPixelBufferLock_ReadOnly);
    CVPixelBufferLockBaseAddress(destinationBuffer, 0);

    // Get buffer addresses and row bytes
    void *sourceBase = CVPixelBufferGetBaseAddress(sourceBuffer);
    void *destBase = CVPixelBufferGetBaseAddress(destinationBuffer);
    size_t sourceRowBytes = CVPixelBufferGetBytesPerRow(sourceBuffer);
    size_t destRowBytes = CVPixelBufferGetBytesPerRow(destinationBuffer);

    // Create vImage buffers
    vImage_Buffer srcBuffer = {
        .data = sourceBase,
        .height = height,
        .width = width,
        .rowBytes = sourceRowBytes
    };

    vImage_Buffer destBuffer = {
        .data = destBase,
        .height = height,
        .width = width,
        .rowBytes = destRowBytes
    };

    // Perform conversion using Accelerate
    vImage_Error convertError = vImageConvert_Planar16FtoPlanarF(&srcBuffer, &destBuffer, 0);
    
    // Unlock buffers
    CVPixelBufferUnlockBaseAddress(sourceBuffer, kCVPixelBufferLock_ReadOnly);
    CVPixelBufferUnlockBaseAddress(destinationBuffer, 0);

    if (convertError != kvImageNoError) {
        NSLog(@"Conversion failed with error: %ld", convertError);
        CFRelease(destinationBuffer);
        return NULL;
    }

    return destinationBuffer; // Caller must release this buffer
}

Channel32fRef DepthHandler::process ( ci::Surface8u& surf ) {
    if(!mInited) return mSurfaceDepth;
    
    if (@available(macOS 14.0, *)) {
        NSError *error = nil;
        
        uint width  = surf.getWidth();
        uint height = surf.getHeight();
        unsigned char *pixels
                    = (unsigned char*)malloc(height*width*4);

        auto pixelIter = surf.getIter();
        int x = 0, y = 0;
        while( pixelIter.line() ) {
            x = 0;
            while( pixelIter.pixel() ) {
                int idx         = (width*y+x)*4;
                pixels[idx+0]   = pixelIter.b();
                pixels[idx+1]   = pixelIter.g();
                pixels[idx+2]   = pixelIter.r();
                pixels[idx+3]   = 255;
                x ++;
            }
            y ++;
        }

        CVPixelBufferRef pxbuffer = NULL;
        CVPixelBufferCreateWithBytes(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_32BGRA,
            (void *)pixels,
            4 * width,
            NULL,
            NULL,
            (__bridge CFDictionaryRef)(attributes),
            &pxbuffer);

        {
            ci::ThreadSetup setup;
            DepthAnythingV2SmallF16Output *output = [_depthModel predictionFromImage:pxbuffer error:&error];
           
            if (error != nil) {
                NSString *err = error.localizedDescription;
                CI_LOG_E("DepthAnything returned with error:" << [err UTF8String]);
            } else {
                // not sure if there are faster ways
                CVPixelBufferLockBaseAddress(output.depth, kCVPixelBufferLock_ReadOnly);
                CVPixelBufferRef buffer32f  = ConvertPixelBufferTo32Float(output.depth);
                CVPixelBufferLockBaseAddress(buffer32f, kCVPixelBufferLock_ReadOnly);
                float* ptr                  = (float*)CVPixelBufferGetBaseAddress(buffer32f);

                int32_t mExposedFrameBytesPerRow = (int32_t)::CVPixelBufferGetBytesPerRow( buffer32f );
                int32_t mExposedFrameWidth       = (int32_t)::CVPixelBufferGetWidth( buffer32f );
                int32_t mExposedFrameHeight      = (int32_t)::CVPixelBufferGetHeight( buffer32f );

                mSurfaceDepth = std::shared_ptr<cinder::Channel32f>(
                    new cinder::Channel32f( mExposedFrameWidth, mExposedFrameHeight, mExposedFrameBytesPerRow, 1, ptr ),
                    [buffer32f]( cinder::Channel32f* s ) {
                        delete s;
                        ::CVPixelBufferUnlockBaseAddress( buffer32f, 0 );
                        ::CVBufferRelease( buffer32f );
                } );
                
                auto it = mSurfaceDepth->getIter();
                mDepthRange = vec2(65535.f, -1.f);
                while(it.line()){
                    while(it.pixel()){
                        mDepthRange.x = glm::min(mDepthRange.x, it.v());
                        mDepthRange.y = glm::max(mDepthRange.y, it.v());
                    }
                }
                
                CVPixelBufferUnlockBaseAddress(output.depth, kCVPixelBufferLock_ReadOnly);
            }
        }
            
        free(pixels);
        CVPixelBufferRelease(pxbuffer);
    }
    
    return mSurfaceDepth;
}
