#ifndef STRUCTS_H
#define STRUCTS_H

#include <opencv2/core/core.hpp>

struct Displacement
{
  int up_fine;
  int down_fine;
  int left_fine;
  int right_fine;
  int up_coarse;
  int down_coarse;
  int left_coarse;
  int right_coarse;
};

struct ImagePyramids
{
  std::vector<cv::Mat> GPA;
  std::vector<cv::Mat> GPA_p;
  std::vector<cv::Mat> GPB;
  std::vector<cv::Mat> GPB_p;
};

#endif
