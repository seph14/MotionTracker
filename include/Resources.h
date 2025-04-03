#pragma once
#include "cinder/CinderResources.h"

#define SHADER_RELOAD   0
#define BODY_TEST       1
#define PRODUCTION      1
#define MOTION_CONTROL  1
#define TRACKING_DEBUG  0
#define PROCESS_DEBUG   1
//#define RES_MY_RES			CINDER_RESOURCE( ../resources/, image_name.png, 128, IMAGE )

#define BLEND_VERT  CINDER_RESOURCE( ../resources/shaders/, blend.vert,    128, GLSL )
#define BLEND_FRAG  CINDER_RESOURCE( ../resources/shaders/, blend.frag,    129, GLSL )
#define UI_FONT     CINDER_RESOURCE( ../resources/, Roboto-Bold.ttf, 130, DATA )




