OCImg
=====

Transplant CImg APIs to OpenCV.

## Done

void threshold(Mat &image, const float value, const bool soft_threshold = false, const bool strict_threshold = false);
void normalize(Mat &image, const float minValue, const float maxValue);

void blur(Mat& image, const float sigma, const bool boundary_conditions=true, const bool is_gaussian=false);
void blur(Mat& image, const float sigma_x, const float sigma_y, const float sigma_z, const bool boundary_conditions=true, const bool is_gaussian=false);

void vanvliet(Mat& image, const float sigma, const int order, const char axis='x', const bool boundary_conditions=true);
void deriche(Mat &image, const float sigma, const int order=0, const char axis='x', const bool boundary_conditions=true);

void _recursive_apply(float *data, const float filter[], int K, const int N, const unsigned long off, const int order, const bool boundary_conditions);
void _deriche_apply(float *ptrX, float *ptrY, float a0, float a1, float a2, float a3, float b1, float b2, float coefp, float coefn, int N, unsigned long off, bool boundary_conditions);

## Under transplant
Mat get_structure_tensors(const Mat image, const unsigned int scheme=2);
void diffusion_tensors(Mat &image, const float sharpness=0.7f, const float anisotropy=0.6f, const float alpha=0.6f, const float sigma=1.1f, const bool is_sqrt=false);
