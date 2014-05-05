#ifndef IMAGE_ANALOGY_H
#define IMAGE_ANALOGY_H
#include "structs.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/flann/flann.hpp>
#include <vector>

class ImageAnalogy
{
  public:
  ImageAnalogy(float i_K = 1, int i_n_size = 5, int i_L = 1) :
    K(i_K), n_size(i_n_size), L(i_L)
  { 
    h_size = n_size / 2;

    cv::Mat gaussian_kernelx_big = cv::getGaussianKernel(2*h_size + 1, -1);
    gaussian_kernel_big = gaussian_kernelx_big * gaussian_kernelx_big.t();
    cv::Mat gaussian_kernelx_small = cv::getGaussianKernel(2*(h_size/2) + 1, -1);
    gaussian_kernel_small = gaussian_kernelx_small * gaussian_kernelx_small.t();
  
    disp_default.up_fine = h_size;     disp_default.up_coarse = h_size/2;
    disp_default.down_fine = h_size;   disp_default.down_coarse = h_size/2;
    disp_default.left_fine = h_size;   disp_default.left_coarse = h_size/2;
    disp_default.right_fine = h_size;  disp_default.right_coarse = h_size/2;
  };

  // given images, A, A_p and B, this function will construct B_p
  void create_image_analogy(const cv::Mat& A, const cv::Mat& A_p, 
                          const cv::Mat& B, cv::Mat& B_p);

  private:
  Displacement disp_default;
  float K;
  int n_size, h_size, L;
  cv::Mat gaussian_kernel_big;
  cv::Mat gaussian_kernel_small;
  //given the four images, find the best match for Pixel q using
  //Approximate Nearest-Neighbor and Coherence searches
  cv::Point best_match(const ImagePyramids& IP,
					   const std::vector<cv::Mat>& PS, 
                       const int level_index,
					   const cv::Point q,
                       cv::flann::Index& kdtree,
                       const cv::Mat& fm);
  
  
  cv::Point brute_force_search(const ImagePyramids& IP,
                               const int level_index, const cv::Point q,
                               const std::vector<float>& FQ,
                               const Displacement disp,
                               float& best_dist);

  cv::Point best_approximate_match(const ImagePyramids& IP,
								   const int level_index, const cv::Point q,
                                   const std::vector<float>& FQ,
                                   const Displacement disp,
                                   float& best_dist,
                                   cv::flann::Index& kdtree,
                                   const cv::Mat& fm);


  cv::Point best_coherence_match(const ImagePyramids& IP,
								 const std::vector<cv::Mat>& PS, 
                                 const int level_index, const cv::Point q,
                                 const std::vector<float>& FQ,
                                 const Displacement disp, float& best_dist);

  void find_displacement(const int rows, const int cols, const int half_size,
                         const int r, const int c, Displacement& disp,
                         const int coarse_flag);

  void create_feature_matrix(const int l, const std::vector<cv::Mat>& GPM,
                             const std::vector<cv::Mat>& GPM_p, 
                             cv::Mat& feature_matrix);

  void create_feature_vector(std::vector<float>& F, const std::vector<cv::Mat>& GPM,
                             const std::vector<cv::Mat>& GPM_p, const int row, 
                             const int col, const int l, const Displacement& disp);

  void construct_gaussian(const cv::Mat& base, std::vector<cv::Mat>& GP);
  float l2_norm(const std::vector<float>& F1, const std::vector<float>& F2);
  void test_flann();
};

#endif
