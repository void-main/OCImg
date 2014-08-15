//
//  Utils.cpp
//  DetailManipulation
//
//  Created by Sun Peng on 14-8-12.
//  Copyright (c) 2014年 NexusHubs. All rights reserved.
//

#include "Utils.h"
#include <math.h>

char OCImg::uncase(const char c)
{
    int result = c;
    if (result >= 65 && result <= 90) {
        result += 32;
    }
    return result;
}

float OCImg::round(float x, float y, const int rounding_type)
{
    const double sx = (double)x / y, floor = floorf(sx), delta =  sx - floor;
    return (y * (rounding_type < 0 ? floor : rounding_type > 0 ? ceilf(sx) : delta < 0.5 ? floor : ceilf(sx)));
}

void OCImg::cut(cv::Mat& image, const float minValue, const float maxValue)
{
    if (image.size().area()) {
        const float a = minValue < maxValue ? minValue : maxValue;
        const float b = minValue < maxValue ? maxValue : minValue;

        image = image * -1;
        cv::threshold(image, image, a, a, cv::THRESH_TRUNC);
        image = image * -1;
        cv::threshold(image, image, b, b, cv::THRESH_TRUNC);
    }
}