#include "ImageAnalogy.h"
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
  Mat A_8u1 = imread(argv[1], -1);
  Mat Ap_8u1 = imread(argv[2], -1);
  Mat B_8u1 = imread(argv[3], -1);
  cvtColor(A_8u1, A_8u1, CV_BGR2GRAY);
  cvtColor(B_8u1, B_8u1, CV_BGR2GRAY);
  cvtColor(Ap_8u1, Ap_8u1, CV_BGR2GRAY);
  int num_levels = 3;
  
  ImageAnalogy IA(2, 5, num_levels);

  namedWindow("A", CV_WINDOW_AUTOSIZE);
  imshow("A", A_8u1); 
  namedWindow("AP", CV_WINDOW_AUTOSIZE);
  imshow("AP", Ap_8u1); 
  namedWindow("B", CV_WINDOW_AUTOSIZE);
  imshow("B", B_8u1); 
  waitKey(0);
  

  Mat BP;
  IA.create_image_analogy(A_8u1, Ap_8u1, B_8u1, BP);
  //imwrite(
 
  std::stringstream ss;
  ss << num_levels << ".jpg";
  std::string filename;
  std::string bn = std::string(argv[3]);
  int pos = bn.find(bn, '.');
  bn = bn.substr(7, pos - 7);
  filename = std::string("Results/coherence_") + bn + std::string("-l") + ss.str();
  //std::cout << filename << std::endl; 
  imwrite(filename, BP);


  namedWindow("BP", CV_WINDOW_AUTOSIZE);
  imshow("BP", BP);
  namedWindow("A", CV_WINDOW_AUTOSIZE);
  imshow("A", A_8u1); 
  namedWindow("AP", CV_WINDOW_AUTOSIZE);
  imshow("AP", Ap_8u1); 
  namedWindow("B", CV_WINDOW_AUTOSIZE);
  imshow("B", B_8u1); 
  waitKey(0);
}
