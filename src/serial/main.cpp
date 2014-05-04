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

  //IMREAD IMAGES MUST BE GRAYSCALE
  Mat A = imread(argv[1], -1);
  Mat Ap = imread(argv[2], -1);
  Mat B = imread(argv[3], -1);
  //cvtColor(A_8u1, A_8u1, CV_BGR2GRAY);
  //cvtColor(B_8u1, B_8u1, CV_BGR2GRAY);
  //cvtColor(Ap_8u1, Ap_8u1, CV_BGR2GRAY);
  
  ColorSpace cs;
  Mat A_YIQ, Ap_YIQ, B_YIQ;
  Mat A_RGB;
  cs.RGB_to_YIQ(A, A_YIQ);
  cs.RGB_to_YIQ(Ap, Ap_YIQ);
  cs.RGB_to_YIQ(B, B_YIQ);
  cs.YIQ_to_RGB(A_YIQ, A_RGB);
  namedWindow("A_RGB", CV_WINDOW_AUTOSIZE);
  imshow("A_RGB", A_RGB); 


  Mat A_YIQ_split[3];
  Mat Ap_YIQ_split[3];
  Mat B_YIQ_split[3];

  split(A_YIQ, A_YIQ_split);
  split(Ap_YIQ, Ap_YIQ_split);
  split(B_YIQ, B_YIQ_split);
  int num_levels = 5;
  
  ImageAnalogy IA(2, 5, num_levels);

  namedWindow("Ay", CV_WINDOW_AUTOSIZE);
  imshow("Ay", A_YIQ_split[0]); 
  namedWindow("APy", CV_WINDOW_AUTOSIZE);
  imshow("APy", Ap_YIQ_split[0]); 
  namedWindow("B", CV_WINDOW_AUTOSIZE);
  imshow("B", B_YIQ_split[0]); 
  waitKey(0);
  

  Mat BP_Y, BP;
  IA.create_image_analogy(A_YIQ_split[0], Ap_YIQ_split[0], B_YIQ_split[0], BP_Y);
  //IA.create_image_analogy(A_8u1, Ap_8u1, B_8u1, BP);
  //imwrite(

  cs.YIQ_to_RGB(BP_Y, B_YIQ, BP);
 
  std::stringstream ss;
  ss << num_levels << ".jpg";
  std::string filename;
  std::string bn = std::string(argv[3]);
  int pos = bn.find(bn, '.');
  bn = bn.substr(7, pos - 7);
  filename = std::string("Results/flann_coh_YIQ") + bn + std::string("-l") + ss.str();
  //std::cout << filename << std::endl; 
  imwrite(filename, BP);


  namedWindow("BP_Y", CV_WINDOW_AUTOSIZE);
  imshow("BP_Y", BP_Y);
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
