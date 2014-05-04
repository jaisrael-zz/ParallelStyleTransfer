#ifndef COLOR_SPACE_H
#define COLOR_SPACE_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

class ColorSpace
{
public:
  void RGB_to_YIQ(const uchar blue, const uchar green, const uchar red,
                  uchar& Y, uchar& I, uchar& Q);
  
  void YIQ_to_RGB(const uchar Y, const uchar I, const uchar Q,
                  uchar& blue, uchar& green, uchar& red);

  void RGB_to_YIQ(const cv::Mat& RGB, cv::Mat& YIQ);
  void YIQ_to_RGB(const cv::Mat& YIQ, cv::Mat& RGB);
  void YIQ_to_RGB(const cv::Mat& Y, const cv::Mat& YIQ, cv::Mat& RGB);

};

void ColorSpace::RGB_to_YIQ(const uchar blue, const uchar green, const uchar red,
                            uchar& Y, uchar& I, uchar& Q)
{
  int iY = (int) (0.299*red + 0.587*green + 0.114*blue);
  int iI = (int) (0.596*red - 0.275*green - 0.321*blue);
  int iQ = (int) (0.212*red - 0.523*green + 0.311*blue);

  if (iY > 255) iY = 255; if (iY < 0) iY = 0;
  if (iI > 255) iI = 255; if (iI < 0) iI = 0;
  if (iQ > 255) iQ = 255; if (iQ < 0) iQ = 0;

  Y = iY; I = iI; Q = iQ;
}

void ColorSpace::YIQ_to_RGB(const uchar Y, const uchar I, const uchar Q,
                            uchar& blue, uchar& green, uchar& red)
{
  int i_red   = (int) (Y + 0.95568806036116*I + 0.61985809445637*Q);
  int i_green = (int) (Y - 0.27158179694406*I - 0.64687381613840*Q);
  int i_blue  = (int) (Y - 1.10817732668266*I + 1.70506455991918*Q);

  if (i_red > 255)   i_red = 255;   if (i_red < 0)   i_red = 0;
  if (i_green > 255) i_green = 255; if (i_green < 0) i_green = 0;
  if (i_blue > 255)  i_blue = 255;  if (i_blue < 0)  i_blue = 0;
  
  red = i_red; green = i_green; blue = i_blue;
  
}

void ColorSpace::RGB_to_YIQ(const cv::Mat& RGB, cv::Mat& YIQ)
{
  YIQ.create(RGB.rows, RGB.cols, RGB.type());
  for (int r = 0; r < RGB.rows; r++)
  {
    const Vec3b* rgb_row_ptr = RGB.ptr<Vec3b>(r);
    Vec3b* yiq_row_ptr = YIQ.ptr<Vec3b>(r);
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

  RGB.create(YIQ.rows, YIQ.cols, YIQ.type());
  for (int r = 0; r < RGB.rows; r++)
  {
    const Vec3b* yiq_row_ptr = YIQ.ptr<Vec3b>(r);
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

  
  RGB.create(YIQ.rows, YIQ.cols, YIQ.type());
  assert(Y.rows == YIQ.rows);
  assert(Y.cols == YIQ.cols);
  

  for (int r = 0; r < RGB.rows; r++)
  {
    const Vec3b* yiq_row_ptr = YIQ.ptr<Vec3b>(r);
    const uchar* y_row_ptr = Y.ptr<uchar>(r);
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
