#ifndef COLOR_SPACE_H
#define COLOR_SPACE_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "math.h"

using namespace cv;

class ColorSpace
{
public:
  void RGB_to_YIQ(const uchar blue, const uchar green, const uchar red,
                  double& Y, double& I, double& Q);
  
  void YIQ_to_RGB(const double Y, const double I, const double Q,
                  uchar& blue, uchar& green, uchar& red);

  void find_mean_stddev(const cv::Mat& Y, double& mean, double& stddev);
  void luminance_remap(const cv::Mat& YA, const cv::Mat& YB, cv::Mat& YA_remapped);

  void RGB_to_YIQ(const cv::Mat& RGB, cv::Mat& YIQ);
  void YIQ_to_RGB(const cv::Mat& YIQ, cv::Mat& RGB);
  void YIQ_to_RGB(const cv::Mat& Y, const cv::Mat& YIQ, cv::Mat& RGB);

};

void ColorSpace::find_mean_stddev(const Mat& Y, double& mean, double& stddev)
{
  int n = Y.rows*Y.cols;
  mean = 0.0;
  for (int r = 0; r < Y.rows; r++)
  {
    const uchar* y_row_ptr = Y.ptr<uchar>(r);
    for (int c = 0; c < Y.cols; c++)
    {
      mean += *y_row_ptr;
      y_row_ptr++;
    }
  }   
  mean = mean / n;

  double sum_squares = 0;

  for (int r = 0; r < Y.rows; r++)
  {
    const uchar* y_row_ptr = Y.ptr<uchar>(r);
    for (int c = 0; c < Y.cols; c++)
    {
      sum_squares += pow(*y_row_ptr - mean,2);
      y_row_ptr++;
    }
  }
  
  stddev = sum_squares / n;
  stddev = sqrt(stddev);
}

void ColorSpace::luminance_remap(const Mat& YA, const Mat& YB, Mat& YA_remapped)
{
  double stddev_b, stddev_a, mean_b, mean_a;
  YA_remapped.create(YA.rows, YA.cols, YA.type());
  //find mean for each
  find_mean_stddev(YA, mean_a, stddev_a);
  find_mean_stddev(YB, mean_b, stddev_b);

  for (int r = 0; r < YA.rows; r++)
  {
    const uchar* y_row_ptr = YA.ptr<uchar>(r);
    uchar* yrm_row_ptr = YA_remapped.ptr<uchar>(r);
    for (int c = 0; c < YA.cols; c++)
    {

      int result = (int) ((stddev_b / stddev_a)*(*y_row_ptr - mean_a) + mean_b);
      if (result > 255) result = 255; if (result < 0) result = 0;
      *yrm_row_ptr = result;

      y_row_ptr++;
      yrm_row_ptr++;
    }
  }
}

void ColorSpace::RGB_to_YIQ(const uchar blue, const uchar green, const uchar red,
                            double& Y, double& I, double& Q)
{
  int iY = (int) (0.299*(double)red + 0.587*(double)green + 0.114*(double)blue);
  I = (0.595716*(double)red - 0.274453*(double)green - 0.321263*(double)blue);
  Q= (0.211456*(double)red - 0.522591*(double)green + 0.311135*(double)blue);

  if (iY > 255) iY = 255; if (iY < 0) iY = 0;

  Y = iY;
}

void ColorSpace::YIQ_to_RGB(const double Y, const double I, const double Q,
                            uchar& blue, uchar& green, uchar& red)
{
  int i_red   = (int) (Y + 0.9563*I + 0.621*Q);
  int i_green = (int) (Y - 0.2721*I - 0.6474*Q);
  int i_blue  = (int) (Y - 1.1070*I + 1.7046*Q);

  if (i_red > 255)   i_red = 255;   if (i_red < 0)   i_red = 0;
  if (i_green > 255) i_green = 255; if (i_green < 0) i_green = 0;
  if (i_blue > 255)  i_blue = 255;  if (i_blue < 0)  i_blue = 0;
  
  red = i_red; green = i_green; blue = i_blue;
  
}

void ColorSpace::RGB_to_YIQ(const cv::Mat& RGB, cv::Mat& YIQ)
{
  YIQ.create(RGB.rows, RGB.cols, CV_64FC3);
  for (int r = 0; r < RGB.rows; r++)
  {
    const Vec3b* rgb_row_ptr = RGB.ptr<Vec3b>(r);
    Vec3d* yiq_row_ptr = YIQ.ptr<Vec3d>(r);
    for (int c = 0; c < RGB.cols; c++)
    {
      RGB_to_YIQ((*rgb_row_ptr)[0], (*rgb_row_ptr)[1], (*rgb_row_ptr)[2],
                 (*yiq_row_ptr)[0], (*yiq_row_ptr)[1], (*yiq_row_ptr)[2]);

      rgb_row_ptr++;
      yiq_row_ptr++;
    }
    
  }
    
}
void ColorSpace::YIQ_to_RGB(const cv::Mat& YIQ, cv::Mat& RGB)
{

  RGB.create(YIQ.rows, YIQ.cols, CV_8UC3);
  for (int r = 0; r < RGB.rows; r++)
  {
    const Vec3d* yiq_row_ptr = YIQ.ptr<Vec3d>(r);
    Vec3b* rgb_row_ptr = RGB.ptr<Vec3b>(r);
    for (int c = 0; c < RGB.cols; c++)
    {
      YIQ_to_RGB((*yiq_row_ptr)[0], (*yiq_row_ptr)[1], (*yiq_row_ptr)[2],
                 (*rgb_row_ptr)[0], (*rgb_row_ptr)[1], (*rgb_row_ptr)[2]);

      rgb_row_ptr++;
      yiq_row_ptr++;
    }
    
  }
}

void ColorSpace::YIQ_to_RGB(const cv::Mat& Y, const cv::Mat& YIQ, cv::Mat& RGB)
{

  
  RGB.create(YIQ.rows, YIQ.cols, CV_8UC3);
  assert(Y.rows == YIQ.rows);
  assert(Y.cols == YIQ.cols);
  

  for (int r = 0; r < RGB.rows; r++)
  {
    const Vec3d* yiq_row_ptr = YIQ.ptr<Vec3d>(r);
    const double* y_row_ptr = Y.ptr<double>(r);
    Vec3b* rgb_row_ptr = RGB.ptr<Vec3b>(r);
    for (int c = 0; c < RGB.cols; c++)
    {
      YIQ_to_RGB((*y_row_ptr), (*yiq_row_ptr)[1], (*yiq_row_ptr)[2],
                 (*rgb_row_ptr)[0], (*rgb_row_ptr)[1], (*rgb_row_ptr)[2]);

      rgb_row_ptr++;
      yiq_row_ptr++;
      y_row_ptr++;
    }
    
  }
}

#endif
