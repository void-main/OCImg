//
//  Utils.h
//  DetailManipulation
//
//  Created by Sun Peng on 14-8-12.
//  Copyright (c) 2014年 NexusHubs. All rights reserved.
//

#ifndef __DetailManipulation__Utils__
#define __DetailManipulation__Utils__

#include <iostream>
#include <opencv2/opencv.hpp>

namespace OCImg {
    char uncase(const char c);
    float round(float x, float y = 1, const int rounding_type = 0);

    void cut(cv::Mat& image, const float minValue, const float maxValue);
}

#endif /* defined(__DetailManipulation__Utils__) */
