#ifndef IMAGE_ANALOGY_H
#define IMAGE_ANALOGY_H
#include "structs.h"
#include <opencv2/core/core.hpp>
#include <vector>

class ImageAnalogy
{
  public:
  ImageAnalogy(double i_K = 1, int i_n_size = 5, int i_L = 1) :
    K(i_K), n_size(i_n_size), L(i_L)
  { 
    h_size = n_size / 2;
  };

  // given images, A, A_p and B, this function will construct B_p
  void create_image_analogy(const cv::Mat& A, const cv::Mat& A_p, 
                          const cv::Mat& B, cv::Mat& B_p);

  private:
  double K;
  int n_size, h_size, L; 
  //given the four images, find the best match for Pixel q using
  //Approximate Nearest-Neighbor and Coherence searches
  cv::Point best_match(const ImagePyramids& IP,
					   const std::vector<cv::Mat>& PS, 
                       const int level_index,
					   const cv::Point q);
  
  
  cv::Point brute_force_search(const ImagePyramids& IP,
                               const int level_index, const cv::Point q,
                               const std::vector<uchar>& FQ,
                               const Displacement disp);

  cv::Point best_approximate_match(const ImagePyramids& IP,
								   const int level_index, const cv::Point q,
                                   const std::vector<uchar>& FQ,
                                   const Displacement disp);


  cv::Point best_coherence_match(const ImagePyramids& IP,
								 const std::vector<cv::Mat>& PS, 
                                 const int level_index, const cv::Point q,
                                 const std::vector<uchar>& FQ,
                                 const Displacement disp);

  void find_displacement(const int rows, const int cols, const int half_size,
                         const int r, const int c, Displacement& disp,
                         const int coarse_flag);

  void create_feature_vector(std::vector<uchar>& F, const std::vector<cv::Mat>& GPM,
                             const std::vector<cv::Mat>& GPM_p, const int row, 
                             const int col, const int l, const Displacement& disp);

  void construct_gaussian(const cv::Mat& base, std::vector<cv::Mat>& GP);
  double l2_norm(const std::vector<uchar>& F1, const std::vector<uchar>& F2);
};

#endif
