//
//  OCImg.cpp
//  DetailManipulation
//
//  Created by Sun Peng on 14-8-12.
//  Copyright (c) 2014年 NexusHubs. All rights reserved.
//

#include "OCImg.h"
#include "Utils.h"

using namespace cv;

void OCImg::threshold(Mat &image, const float value, const bool soft_threshold, const bool strict_threshold)
{
    if (image.size().area()) {
        if (strict_threshold) {
            if (soft_threshold) {
                Mat upper = (image - value);
                Mat lower = (image + value);
                cv::threshold(upper, upper, 0, 0, cv::THRESH_TOZERO);
                cv::threshold(lower, lower, 0, 0, cv::THRESH_TOZERO_INV);
                image = upper + lower;
            }
            else {
                cv::threshold(image, image, value, 1.0, cv::THRESH_BINARY);
            }
        } else { // Non-strict, allows `=`.
            if (soft_threshold) {
                Mat upper = (image - value);
                Mat lower = (image + value);
                cv::threshold(upper, upper, 0, 0, cv::THRESH_TOZERO);
                cv::threshold(lower, lower, 0, 0, cv::THRESH_TOZERO_INV);
                image = upper + lower;
            }
            else {
                Mat eqal = (image == value);
                cv::threshold(image, image, value, 1.0, cv::THRESH_BINARY);
                image = image + eqal;
            }
        }
    }
}

void normalize(Mat &image, const float minValue, const float maxValue)
{
    if (image.size().area()) {
        const float a = minValue < maxValue ? minValue : maxValue;
        const float b = minValue < maxValue ? maxValue : minValue;

        double imageMin, imageMax;
        cv::minMaxLoc(image, &imageMin, &imageMax);

        if (imageMax == imageMax) {
            image = cv::Mat(image.rows, image.cols, image.type(), Scalar::all(minValue));
        } else if (a != imageMin && b != imageMax) {
            image = (b - a) * (image - imageMin) / (imageMax - imageMin) + a;
        }
    }
}

//#define CImg_3x3(I,T) T I[9]; \
//T& I##pp = I[0]; T& I##cp = I[1]; T& I##np = I[2]; \
//T& I##pc = I[3]; T& I##cc = I[4]; T& I##nc = I[5]; \
//T& I##pn = I[6]; T& I##cn = I[7]; T& I##nn = I[8]; \
//I##pp = I##cp = I##np = \
//I##pc = I##cc = I##nc = \
//I##pn = I##cn = I##nn = 0
//
//#define cimg_for3x3x3(img,x,y,z,c,I,T) \
//cimg_for3((img)._depth,z) cimg_for3((img)._height,y) for (int x = 0, \
//_p1##x = 0, \
//_n1##x = (int)( \
//(I[0] = I[1] = (T)(img)(_p1##x,_p1##y,_p1##z,c)), \
//(I[3] = I[4] = (T)(img)(0,y,_p1##z,c)),  \
//(I[6] = I[7] = (T)(img)(0,_n1##y,_p1##z,c)), \
//(I[9] = I[10] = (T)(img)(0,_p1##y,z,c)), \
//(I[12] = I[13] = (T)(img)(0,y,z,c)), \
//(I[15] = I[16] = (T)(img)(0,_n1##y,z,c)), \
//(I[18] = I[19] = (T)(img)(0,_p1##y,_n1##z,c)), \
//(I[21] = I[22] = (T)(img)(0,y,_n1##z,c)), \
//(I[24] = I[25] = (T)(img)(0,_n1##y,_n1##z,c)), \
//1>=(img)._width?(img).width()-1:1); \
//(_n1##x<(img).width() && ( \
//(I[2] = (T)(img)(_n1##x,_p1##y,_p1##z,c)), \
//(I[5] = (T)(img)(_n1##x,y,_p1##z,c)), \
//(I[8] = (T)(img)(_n1##x,_n1##y,_p1##z,c)), \
//(I[11] = (T)(img)(_n1##x,_p1##y,z,c)), \
//(I[14] = (T)(img)(_n1##x,y,z,c)), \
//(I[17] = (T)(img)(_n1##x,_n1##y,z,c)), \
//(I[20] = (T)(img)(_n1##x,_p1##y,_n1##z,c)), \
//(I[23] = (T)(img)(_n1##x,y,_n1##z,c)), \
//(I[26] = (T)(img)(_n1##x,_n1##y,_n1##z,c)),1)) || \
//x==--_n1##x; \
//I[0] = I[1], I[1] = I[2], I[3] = I[4], I[4] = I[5], I[6] = I[7], I[7] = I[8], \
//I[9] = I[10], I[10] = I[11], I[12] = I[13], I[13] = I[14], I[15] = I[16], I[16] = I[17], \
//I[18] = I[19], I[19] = I[20], I[21] = I[22], I[22] = I[23], I[24] = I[25], I[25] = I[26], \
//_p1##x = x++, ++_n1##x)
//
//
//Mat OCImg::get_structure_tensors(const Mat image, const unsigned int scheme) {
//    if (image.size().area()) {
//        Mat res(image.rows, image.cols, image.type(), cv::Scalar::all(0));
//        vector<Mat> planes;
//        split(res, planes);
//
//        for (int idx = 0; idx < planes.size(); idx ++) {
//            switch (scheme) {
//                case 0 : { // classical central finite differences
//                    Tfloat *ptrd0 = res.data(0,0,0,0), *ptrd1 = res.data(0,0,0,1), *ptrd2 = res.data(0,0,0,2);
//                    CImg_3x3(I,Tfloat);
//                    cimg_for3x3(*this,x,y,0,c,I,Tfloat) {
//                        const Tfloat
//                        ix = (Inc - Ipc)/2,
//                        iy = (Icn - Icp)/2;
//                        *(ptrd0++)+=ix*ix;
//                        *(ptrd1++)+=ix*iy;
//                        *(ptrd2++)+=iy*iy;
//                    }
//                } break;
//                case 1 : { // Forward/backward finite differences (version 1).
//                    Tfloat *ptrd0 = res.data(0,0,0,0), *ptrd1 = res.data(0,0,0,1), *ptrd2 = res.data(0,0,0,2);
//                    CImg_3x3(I,Tfloat);
//                    cimg_for3x3(*this,x,y,0,c,I,Tfloat) {
//                        const Tfloat
//                        ixf = Inc - Icc, ixb = Icc - Ipc,
//                        iyf = Icn - Icc, iyb = Icc - Icp;
//                        *(ptrd0++)+=(ixf*ixf + 2*ixf*ixb + ixb*ixb)/4;
//                        *(ptrd1++)+=(ixf*iyf + ixf*iyb + ixb*iyf + ixb*iyb)/4;
//                        *(ptrd2++)+=(iyf*iyf + 2*iyf*iyb + iyb*iyb)/4;
//                    }
//                } break;
//                default : { // Forward/backward finite differences (version 2).
//                    Tfloat *ptrd0 = res.data(0,0,0,0), *ptrd1 = res.data(0,0,0,1), *ptrd2 = res.data(0,0,0,2);
//                    CImg_3x3(I,Tfloat);
//                    cimg_for3x3(*this,x,y,0,c,I,Tfloat) {
//                        const Tfloat
//                        ixf = Inc - Icc, ixb = Icc - Ipc,
//                        iyf = Icn - Icc, iyb = Icc - Icp;
//                        *(ptrd0++)+=(ixf*ixf + ixb*ixb)/2;
//                        *(ptrd1++)+=(ixf*iyf + ixf*iyb + ixb*iyf + ixb*iyb)/4;
//                        *(ptrd2++)+=(iyf*iyf + iyb*iyb)/2;
//                    }
//                } break;
//            }
//        }
//
//        merge(planes, res);
//        return res;
//    }
//}
//
//void OCImg::diffusion_tensors(Mat &image, const float sharpness, const float anisotropy, const float alpha, const float sigma, const bool is_sqrt)
//{
//    Mat res;
//    const float nsharpness = fmaxf(sharpness, 1e-5f);
//    const float power1 = (is_sqrt ? 0.5f : 1) * nsharpness;
//    const float power2 = power1 / (1e-7f + 1 - anisotropy);
//
//    blur(image, alpha);
//    normalize(image, 0, 255);
//
//    get_structure_tensors().move_to(res).blur(sigma);
//    cimg_forY(*this,y) {
//        Tfloat *ptrd0 = res.data(0,y,0,0), *ptrd1 = res.data(0,y,0,1), *ptrd2 = res.data(0,y,0,2);
//        CImg<floatT> val(2), vec(2,2);
//        cimg_forX(*this,x) {
//            res.get_tensor_at(x,y).symmetric_eigen(val,vec);
//            const float
//            _l1 = val[1], _l2 = val[0],
//            l1 = _l1>0?_l1:0, l2 = _l2>0?_l2:0,
//            ux = vec(1,0), uy = vec(1,1),
//            vx = vec(0,0), vy = vec(0,1),
//            n1 = (float)std::pow(1+l1+l2,-power1),
//            n2 = (float)std::pow(1+l1+l2,-power2);
//            *(ptrd0++) = n1*ux*ux + n2*vx*vx;
//            *(ptrd1++) = n1*ux*uy + n2*vx*vy;
//            *(ptrd2++) = n1*uy*uy + n2*vy*vy;
//        }
//    }
//
//    res.copyTo(image);
//}

void OCImg::blur(Mat& image, const float sigma, const bool boundary_conditions, const bool is_gaussian)
{
    const float nsigma = sigma >= 0 ? sigma : -sigma * fmaxf(image.cols, fmaxf(image.rows, image.channels())) / 100;
    return blur(image, nsigma,nsigma,nsigma,boundary_conditions,is_gaussian);
}

void OCImg::blur(Mat& image, const float sigma_x, const float sigma_y, const float sigma_z, const bool boundary_conditions, const bool is_gaussian)
{
    assert((image.type() & CV_32F) == CV_32F);

    if (image.size().area()) {
        if (is_gaussian) {
            if (image.cols       > 1) vanvliet(image, sigma_x,0,'x',boundary_conditions);
            if (image.rows       > 1) vanvliet(image, sigma_y,0,'y',boundary_conditions);
            if (image.channels() > 1) vanvliet(image, sigma_z,0,'z',boundary_conditions);
        } else {
            if (image.cols       > 1) deriche(image, sigma_x,0,'x',boundary_conditions);
            if (image.rows       > 1) deriche(image, sigma_y,0,'y',boundary_conditions);
            if (image.channels() > 1) deriche(image, sigma_z,0,'z',boundary_conditions);
        }
    }
}

void OCImg::vanvliet(Mat& image, const float sigma, const int order, const char axis, const bool boundary_conditions) {
    if (!image.size().area()) {
        return;
    }

    const char naxis = OCImg::uncase(axis);
    const float nsigma = sigma >= 0 ? sigma : -sigma * (naxis == 'x' ? image.cols : naxis == 'y' ? image.rows : image.channels()) / 100;

    if (!image.size().area() || (nsigma < 0.1f && !order))
        return;

    const float nnsigma = nsigma < 0.1f ? 0.1f : nsigma,
    q = (float)(nnsigma < 2.5 ? 3.97156-4.14554 * std::sqrt(1 - 0.2689 * nnsigma) : 0.98711 * nnsigma - 0.96330),
    b0 = 1.57825f + 2.44413f * q + 1.4281f * q * q + 0.4222205f * q * q * q,
    b1 = (2.44413f * q + 2.85619f * q * q + 1.26661f * q * q * q),
    b2 = -(1.4281f * q * q + 1.26661f * q *q * q),
    b3 = 0.4222205f * q * q * q,
    B = 1.f - (b1 + b2 + b3) / b0;
    float filter[4];
    filter[0] = B; filter[1] = b1/b0; filter[2] = b2/b0; filter[3] = b3/b0;

    // TODO
    //
    //
    //
    //
    // Error Prone, not tested yet
    //
    //
    //
    //
    //
    switch (naxis) {
        case 'x' : {
            for (int y = 0; y < image.rows; y ++) {
                OCImg::_recursive_apply((float *)image.ptr(y, 0), filter, 4, image.cols, 1U, order, boundary_conditions);
            }
        } break;
        case 'y' : {
            for (int x = 0; x < image.cols; x ++) {
                OCImg::_recursive_apply((float *)image.ptr(0, x), filter, 4, image.rows, (unsigned long)image.cols, order, boundary_conditions);
            }
        } break;
    }
}

void OCImg::_recursive_apply(float *data, const float filter[], int K, const int N, const unsigned long off, const int order, const bool boundary_conditions) {
    float val[K];  // res[n,n-1,n-2,n-3,..] or res[n,n+1,n+2,n+3,..]
    switch (order) {
        case 0 : {
            for (int pass = 0; pass < 2; ++pass) {
                for (int k = 1; k < K; ++k) val[k] = (float)(boundary_conditions ? *data : 0);
                for (int n = 0; n < N; ++n) {
                    val[0] = (float)(*data) * filter[0];
                    for (int k = 1; k < K; ++k) val[0] += val[k] * filter[k];
                    *data = (float)val[0];
                    if (!pass) data += off; else data -= off;
                    for (int k = K-1; k>0; --k) val[k] = val[k-1];
                }
                if (!pass) data-=off;
            }
        } break;
        case 1 : {
            float x[3]; // [front,center,back]
            for (int pass = 0; pass < 2; ++pass) {
                for (int k = 0; k < 3; ++k) x[k] = (float)(boundary_conditions ? *data : 0);
                for (int k = 0; k < K; ++k) val[k] = 0;
                for (int n = 0; n < N - 1; ++n) {
                    if (!pass) {
                        x[0] = (float)*(data + off);
                        val[0] = 0.5f * (x[0] - x[2]) * filter[0];
                    } else val[0] = (float)(*data) * filter[0];
                    for (int k = 1; k < K; ++k) val[0] += val[k] * filter[k];
                    *data = (float)val[0];
                    if (!pass) {
                        data+=off;
                        for (int k = 2; k>0; --k) x[k] = x[k-1];
                    } else data-=off;
                    for (int k = K - 1; k > 0; --k) val[k] = val[k - 1];
                }
                *data = (float)0;
            }
        } break;
        case 2: {
            float x[3]; // [front,center,back]
            for (int pass = 0; pass<2; ++pass) {
                for (int k = 0; k < 3; ++k) x[k] = (float)(boundary_conditions ? *data : 0);
                for (int k = 0; k < K; ++k) val[k] = 0;
                for (int n = 0; n < N - 1; ++n) {
                    if (!pass) { x[0] = (float)*(data + off); val[0] = (x[1] - x[2]) * filter[0]; }
                    else { x[0] = (float)*(data - off); val[0] = (x[2] - x[1]) * filter[0]; }
                    for (int k = 1; k < K; ++k) val[0] += val[k] * filter[k];
                    *data = (float)val[0];
                    if (!pass) data += off; else data -= off;
                    for (int k = 2; k > 0; --k) x[k] = x[k - 1];
                    for (int k = K - 1; k>0; --k) val[k] = val[k - 1];
                }
                *data = (float)0;
            }
        } break;
        case 3: {
            float x[3]; // [front,center,back]
            for (int pass = 0; pass<2; ++pass) {
                for (int k = 0; k < 3; ++k) x[k] = (float)(boundary_conditions ? *data : 0);
                for (int k = 0; k < K; ++k) val[k] = 0;
                for (int n = 0; n < N - 1; ++n) {
                    if (!pass) { x[0] = (float)*(data + off); val[0] = (x[0] - 2 * x[1] + x[2]) * filter[0]; }
                    else { x[0] = (float)*(data - off); val[0] = 0.5f * (x[2] - x[0]) * filter[0]; }
                    for (int k = 1; k < K; ++k) val[0] += val[k] * filter[k];
                    *data = (float)val[0];
                    if (!pass) data += off; else data -= off;
                    for (int k = 2; k > 0; --k) x[k] = x[k - 1];
                    for (int k = K - 1; k > 0; --k) val[k] = val[k - 1];
                }
                *data = (float)0;
            }
        } break;
    }
}

void OCImg::_deriche_apply(float *ptrX, float *ptrY, float a0, float a1, float a2, float a3, float b1, float b2, float coefp, float coefn, int N, unsigned long off, bool boundary_conditions)
{
    float yb = 0, yp = 0;
    float xp = 0;
    if (boundary_conditions) {
        xp = *ptrX; yb = yp = (float)(coefp * xp);
    }

    for (int m = 0; m<N; ++m) {
        const float xc = *ptrX; ptrX += off;
        const float yc = *(ptrY++) = (float)(a0 * xc + a1 * xp - b1 * yp - b2 * yb);
        xp = xc; yb = yp; yp = yc;
    }
    float xn = (float)0, xa = (float)0;
    float yn = 0, ya = 0;
    if (boundary_conditions) { xn = xa = *(ptrX-off); yn = ya = (float)coefn * xn; }
    for (int n = N - 1; n>=0; --n) {
        const float xc = *(ptrX -= off);
        const float yc = (float)(a2 * xn + a3 * xa - b1 * yn - b2 * ya);
        xa = xn; xn = xc; ya = yn; yn = yc;
        *ptrX = (float)(*(--ptrY)+yc);
    }
}

void OCImg::deriche(Mat &image, const float sigma, const int order, const char axis, const bool boundary_conditions) {
    const char naxis = OCImg::uncase(axis);
    const float nsigma = sigma>=0 ? sigma : -sigma * (naxis == 'x' ? image.cols : naxis == 'y' ? image.rows : image.channels())/100;
    if (!image.size().area() || (nsigma < 0.1f && !order))
        return;

    const float
    nnsigma = nsigma<0.1f?0.1f:nsigma,
    alpha = 1.695f/nnsigma,
    ema = (float)std::exp(-alpha),
    ema2 = (float)std::exp(-2*alpha),
    b1 = -2*ema,
    b2 = ema2;
    float a0 = 0, a1 = 0, a2 = 0, a3 = 0, coefp = 0, coefn = 0;
    switch (order) {
        case 0 : {
            const float k = (1-ema)*(1-ema)/(1+2*alpha*ema-ema2);
            a0 = k;
            a1 = k*(alpha-1)*ema;
            a2 = k*(alpha+1)*ema;
            a3 = -k*ema2;
        } break;
        case 1 : {
            const float k = -(1-ema)*(1-ema)*(1-ema)/(2*(ema+1)*ema);
            a0 = a3 = 0;
            a1 = k*ema;
            a2 = -a1;
        } break;
        case 2 : {
            const float
            ea = (float)std::exp(-alpha),
            k = -(ema2-1)/(2*alpha*ema),
            kn = (-2*(-1+3*ea-3*ea*ea+ea*ea*ea)/(3*ea+1+3*ea*ea+ea*ea*ea));
            a0 = kn;
            a1 = -kn*(1+k*alpha)*ema;
            a2 = kn*(1-k*alpha)*ema;
            a3 = -kn*ema2;
        } break;
        default :
            throw Exception();
    }
    coefp = (a0+a1)/(1+b1+b2);
    coefn = (a2+a3)/(1+b1+b2);
    switch (naxis) {
        case 'x' : {
            const int N = image.cols;
            const unsigned long off = 1U;
            Mat Y(1, N, image.type());
            for (int y = 0; y < image.rows; y ++) {
                float *ptrX = (float *)image.ptr(y, 0);
                float *ptrY = (float *)Y.ptr();
                _deriche_apply(ptrX, ptrY, a0, a1, a2, a3, b1, b2, coefp, coefn, N, off, boundary_conditions);
            }
        } break;
        case 'y' : {
            const int N = image.rows;
            const unsigned long off = (unsigned long)image.cols;
            Mat Y(1, N, image.type());
            for (int x = 0; x < image.cols; x ++) {
                float *ptrX = (float *)image.ptr(0, x);
                float *ptrY = (float *)Y.ptr();
                _deriche_apply(ptrX, ptrY, a0, a1, a2, a3, b1, b2, coefp, coefn, N, off, boundary_conditions);
            }
        } break;
    }
}
