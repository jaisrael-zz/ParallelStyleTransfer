#include "ImageAnalogy.h"
#include "ColorSpace.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
using namespace cv;

int main(int argc, char** argv)
{
  if (argc != 4)
  {
    std::cout << "Usage: stylexfer A A' B" << std::endl;
    return -1;
  }

  Mat A = imread(argv[1], -1);
  Mat Ap = imread(argv[2], -1);
  Mat B = imread(argv[3], -1);
  
  ColorSpace cs;
  Mat A_YIQ, Ap_YIQ, B_YIQ;
  Mat A_RGB;
  //RGB TO YIQ
  cs.RGB_to_YIQ(A, A_YIQ);
  cs.RGB_to_YIQ(Ap, Ap_YIQ);
  cs.RGB_to_YIQ(B, B_YIQ);
  cs.YIQ_to_RGB(A_YIQ, A_RGB);
  namedWindow("A_RGB_orig", CV_WINDOW_AUTOSIZE);
  imshow("A_RGB_orig", A); 
  namedWindow("A_RGB_remapped", CV_WINDOW_AUTOSIZE);
  imshow("A_RGB_remapped", A_RGB); 


  Mat A_YIQ_split[3];
  Mat Ap_YIQ_split[3];
  Mat B_YIQ_split[3];

  split(A_YIQ, A_YIQ_split);
  split(Ap_YIQ, Ap_YIQ_split);
  split(B_YIQ, B_YIQ_split);
  int num_levels = 6;

  Mat A_YIQ_8u, Ap_YIQ_8u, B_YIQ_8u;
  A_YIQ_split[0].convertTo(A_YIQ_8u, CV_8UC1); 
  Ap_YIQ_split[0].convertTo(Ap_YIQ_8u, CV_8UC1); 
  B_YIQ_split[0].convertTo(B_YIQ_8u, CV_8UC1); 
  Mat A_Y_RM = A_YIQ_8u; 
  Mat Ap_Y_RM = Ap_YIQ_8u; 
  ImageAnalogy IA(0, 5, num_levels);
  cs.luminance_remap(A_YIQ_8u, B_YIQ_8u, A_Y_RM);
  cs.luminance_remap(Ap_YIQ_8u, B_YIQ_8u, Ap_Y_RM);
  waitKey(0);
  

  Mat BP_Y, BP;
  clock_t start = clock();
  IA.create_image_analogy(A_Y_RM, Ap_Y_RM, B_YIQ_8u, BP_Y);
  //IA.create_image_analogy(A_8u1, Ap_8u1, B_8u1, BP);
  //imwrite(

  Mat BP_Y_64f;
  BP_Y.convertTo(BP_Y_64f, CV_64FC1); 
  cs.YIQ_to_RGB(BP_Y_64f, B_YIQ, BP);
 
  std::cout << "Execution Time: " << double(clock() - start) / (double)CLOCKS_PER_SEC << "s" << std::endl;

  std::stringstream ss;
  ss << num_levels << ".jpg";
  std::string filename;
  std::string bn = std::string(argv[3]);
  int pos = bn.find(bn, '.');
  bn = bn.substr(7, pos - 7);
  filename = std::string("Results/ALL_FLANN") + bn + std::string("-l") + ss.str();
  //std::cout << filename << std::endl; 
  imwrite(filename, BP);


  namedWindow("BP", CV_WINDOW_AUTOSIZE);
  imshow("BP", BP);
  namedWindow("A", CV_WINDOW_AUTOSIZE);
  imshow("A", A); 
  namedWindow("AP", CV_WINDOW_AUTOSIZE);
  imshow("AP", Ap); 
  namedWindow("B", CV_WINDOW_AUTOSIZE);
  imshow("B", B); 
  waitKey(0);
}
