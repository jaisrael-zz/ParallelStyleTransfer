#include "ImageAnalogy.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/flann/flann.hpp>
#include <algorithm>
#include "math.h"
#include "structs.h"
#include <iostream>
using namespace std;
using namespace cv;

void ImageAnalogy::construct_gaussian(const Mat& base, vector<Mat>& GP)
{
  GP.clear();
  Mat temp;
  GP.push_back(base);
  for (int i = 1; i < L; i++)
  {
    pyrDown(GP[i-1], temp);
    GP.push_back(temp);
  }

}

float ImageAnalogy::l2_norm(const vector<float>& F1, const vector<float>& F2)
{
  float sum = 0;
  for (int i = 0; i < F1.size(); i++)
  {
    sum += pow(F1[i] - F2[i],2);
  }
  return sum;
}

void ImageAnalogy::create_feature_vector(vector<float>& F, 
                                         const vector<Mat>& GPM,
                                         const vector<Mat>& GPM_p,
                                         int row, int col, int l,
                                         const Displacement& disp)
{
  //TODO: SCALING AND COARSE LEVEL!!
  int up_displace = disp.up_fine;
  int down_displace = disp.down_fine;
  int left_displace = disp.left_fine;
  int right_displace = disp.right_fine;
  
  F.clear();

  int prime_flag = 0;
  for (int r = (row - up_displace), i = 0; r <= (row + down_displace); r++, i++)
  {
    const uchar* mp_row_ptr = GPM_p[l].ptr<uchar>(r);
    const uchar* m_row_ptr  = GPM[l].ptr<uchar>(r);
    const float* g_row_ptr = gaussian_kernel_big.ptr<float>(i);
    mp_row_ptr += (col - left_displace);
    m_row_ptr += (col - left_displace);
    for (int c = col-left_displace; c <= right_displace+col; c++)
    {
      if (r == row && c == col) prime_flag = 1;
      F.push_back((*m_row_ptr) * (*g_row_ptr));
      if (!prime_flag) F.push_back((*mp_row_ptr) * (*g_row_ptr));
      mp_row_ptr++;
      m_row_ptr++;
      g_row_ptr++;
    }
  }
  //return;
  if (l == L-1) return;

  //COARSE LEVEL!!!!!!
  row = row / 2;
  col = col / 2;
  up_displace = disp.up_coarse;
  down_displace = disp.down_coarse;
  left_displace = disp.left_coarse;
  right_displace = disp.right_coarse;
  //have to add previous level to feature vector as well
  
  for (int r = (row - up_displace), i=0; r <= (row + down_displace); r++, i++)
  {
    const uchar* mp_row_ptr = GPM_p[l+1].ptr<uchar>(r);
    const uchar* m_row_ptr  = GPM[l+1].ptr<uchar>(r);
    const float* g_row_ptr = gaussian_kernel_small.ptr<float>(i);
    mp_row_ptr += (col - left_displace);
    m_row_ptr += (col - left_displace);
    for (int c = col-left_displace; c <= right_displace+col; c++)
    {
      //if (r == row && c == col) prime_flag = 1;
      F.push_back((*m_row_ptr) + (*g_row_ptr));
      F.push_back((*mp_row_ptr) + (*g_row_ptr));
      mp_row_ptr++;
      m_row_ptr++;
      g_row_ptr++;
    }
  }
}
                                         

Point ImageAnalogy::brute_force_search(const ImagePyramids& IP,
                                       const int l, const Point q,
                                       const vector<float>& FQ,
                                       const Displacement disp,
                                       float& best_dist)
{

  //cout << "bfs!" << endl;

  //construct feature vector from B/B'
  //create_feature_vector(FQ, GPB, GPB_p, q_row, q_col, l, up_displace,
  //                      down_displace, left_displace, right_displace); 
  
  int up_displace = disp.up_fine;
  int down_displace = disp.down_fine;
  int left_displace = disp.left_fine;
  int right_displace = disp.right_fine;
 
  vector<float> best_fp; 
  //FQ has now been constructed, search through A/A' for the best pixel
  best_dist = 1E+37;
  Point best_index;
  for (int r = up_displace; r < IP.GPA[l].rows-down_displace; r++)
  {
    for (int c = left_displace; c < IP.GPA[l].cols-right_displace; c++)
    {
      vector<float> FP;
      //construct feature vector from A/A'
      create_feature_vector(FP, IP.GPA, IP.GPA_p, r, c, l, disp);
      //TODO: COARSE LEVEL!!!!!!!
      //compare with FQ
      float norm = l2_norm(FP, FQ);
      //keep track if its the smallest
      if (norm < best_dist)
      {
        //best_score = norm;
        best_index.x = c;
        best_index.y = r;
        best_dist = norm;
        best_fp = FP;
      }
    }
  }
/*  if (l != L-1) {
    cout << "BF-FP = "; 
    for (int i = 0; i < best_fp.size(); i++) cout << best_fp[i] << " ";
    cout << endl;
  }*/
  return best_index;
}

Point ImageAnalogy::best_approximate_match(const ImagePyramids& IP,
                                           int l, Point q,
                                           const vector<float>& FQ,
                                           const Displacement disp,
                                           float& best_dist,
                                           flann::Index& kdtree,
                                           const Mat& feature_matrix)
{
  //Point pb;
  //pb =  brute_force_search(IP, l, q, FQ, disp, best_dist);

  if (disp.up_fine != h_size   || disp.down_fine != h_size ||
      disp.left_fine != h_size || disp.right_fine != h_size)
    return brute_force_search(IP, l, q, FQ, disp, best_dist);
  if (l == L-1) 
    return brute_force_search(IP, l, q, FQ, disp, best_dist);

  //otherwise, do (fl)ANN to figure out best approximate match

  //cout << "PB: " << pb;
  //int brute_index = (pb.y - h_size)*(IP.GPA[l].rows - 2*h_size) +
  //                  (pb.x - h_size);

  //cout << "\tbrute_index: " << brute_index;

  vector<int> index(1);
  vector<float> dist(1);
  
  

  kdtree.knnSearch(FQ, index, dist, 1, flann::SearchParams(32));
   
  Point p;
  p.x = (index[0] % (IP.GPA[l].cols - 2*h_size)) + h_size;
  p.y = (index[0] / (IP.GPA[l].cols - 2*h_size)) + h_size;

  best_dist = dist[0];
  //cout << "\tPNN: " << p;
  //cout << "\tindexknn: " << index[0];
/*
  vector<float> best_fp;
  int best_index;
  float bf_best_dist = 1E37;
  for (int i = 0; i < feature_matrix.rows; i++)
  {
    vector<float> FP;
    feature_matrix.row(i).copyTo(FP);
    float norm = l2_norm(FQ,FP);  
    if (norm < bf_best_dist)
    {
      bf_best_dist = norm;
      best_index = i;
      best_fp = FP;
    }
  }
  cout <<"\tindexmyway: " << best_index << endl;
  cout << "MYWAYFP = "; 
  for (int i = 0; i < best_fp.size(); i++) cout << best_fp[i] << " ";
  cout << "----------------------------";*/
 // }
  //cout << endl;
  return p;
}

Point ImageAnalogy::best_coherence_match(const ImagePyramids& IP,
                                         const vector<Mat>& PS, 
                                         const int l, const Point q,
                                         const vector<float>& FQ,
                                         const Displacement disp,
                                         float& best_dist)
{
  int q_row = q.y;
  int q_col = q.x;
  
  int up_displace = disp.up_fine;
  int down_displace = disp.down_fine;
  int left_displace = disp.left_fine;
  int right_displace = disp.right_fine;
  
  best_dist = 1E+37;
  Point best_index;
  best_index.x = -1;
  best_index.y = -1;
  int never_happens = 0;
  for (int r = q_row-up_displace; r <= q_row; r++)
  {
    const Vec2s* s_row_ptr = PS[l].ptr<Vec2s>(r);
    for (int c = q_col - left_displace; c <= q_col + right_displace; c++)
    {
      if (c >= q_col && r >= q_row) break;
      
      //make sure that s(t) is in bounds, if so construct feature vector,
      //if not continue (move on)
      Point t;
      t.x = (*s_row_ptr)[1];
      t.y = (*s_row_ptr)[0];

      Point t_q;
      t_q.x = t.x + q.x - r;
      t_q.y = t.y + q.y - c;

      //bounds check on t_q
      if (t_q.x < (IP.GPA[l].cols - h_size) && t_q.x > (h_size - 1) &&
          t_q.y < (IP.GPA[l].rows - h_size) && t_q.y > (h_size - 1))
      {
        vector<float> FT_q;
        create_feature_vector(FT_q, IP.GPA, IP.GPA_p, t_q.y, t_q.x, l, disp);
        float norm = l2_norm(FT_q, FQ);
        //keep track if its the smallest
        if (norm < best_dist)
        {
          never_happens = 1;
          best_dist = norm;
          best_index.x = t_q.x;
          best_index.y = t_q.y;
        }
      }

      s_row_ptr++;

    }
  }
  //if (!never_happens) cout << "the impossible happened" << endl;
  return best_index;
}

void ImageAnalogy::find_displacement(const int rows, const int cols,
                                     const int half_size, const int r, const int c,
                                     Displacement& disp, const int coarse_flag)
{

  int start_row = r - half_size;
  if (start_row < 0) start_row = 0;
  int up_displace = r - start_row;
  
  int end_row   = r + half_size;
  if (end_row > (rows - 1)) end_row = (rows - 1);
  int down_displace = end_row - r;
  
  int start_col = c - half_size;
  if (start_col < 0) start_col = 0;
  int left_displace = c - start_col;
  
  int end_col   = c + half_size;
  if (end_col > (cols - 1)) end_col = (cols - 1);
  int right_displace = end_col - c;

  if (coarse_flag)
  {
    disp.up_coarse    = up_displace;
    disp.down_coarse  = down_displace;
    disp.left_coarse  = left_displace;
    disp.right_coarse = right_displace;
  }
  else
  {
    disp.up_fine    = up_displace;
    disp.down_fine  = down_displace;
    disp.left_fine  = left_displace;
    disp.right_fine = right_displace;
  }

  
}

Point ImageAnalogy::best_match(const ImagePyramids& IP, const vector<Mat>& PS, 
                               const int l, const Point q, flann::Index& kdtree,
                               const Mat& feature_matrix)
{
  int q_row = q.y;
  int q_col = q.x;


  Displacement disp;
  find_displacement(IP.GPB[l].rows, IP.GPB[l].cols, h_size, q_row, q_col, disp, 0);
  

  if (l != L-1)
  {
    find_displacement(IP.GPB[l+1].rows, IP.GPB[l+1].cols, h_size/2, 
                      q_row/2, q_col/2, disp, 1);
  }
  
  
  vector<float> FQ;
  float approx_dist, coher_dist; 
  //construct feature vector from B/B'
  create_feature_vector(FQ, IP.GPB, IP.GPB_p, q_row, q_col, l, disp); 
  Point p_a  = best_approximate_match(IP, l, q, FQ, disp, approx_dist, kdtree, feature_matrix);
  if (l == L-1) return p_a;
  
  Point p_c = best_coherence_match(IP, PS, l, q, FQ, disp, coher_dist);
  if (p_c.x == -1) return p_a;
  if (coher_dist <= approx_dist*(1 + pow(2, -l)*K))
    return p_c;
  else
    return p_a;
}

void ImageAnalogy::create_feature_matrix(const int l, const vector<Mat>& GPM,
                                         const vector<Mat>& GPM_p, 
                                         Mat& feature_matrix)
{

  if (l == L-1) return;
  int inner_rows = GPM[l].rows - 2*h_size;
  int inner_cols = GPM[l].cols - 2*h_size;
  //int num_dimensions = (n_size*n_size) + (n_size*n_size)/2;

  int num_dimensions = (n_size*n_size) + (n_size*n_size)/2 + 
                           2*(1+h_size)*(1+h_size);
  feature_matrix.create(inner_rows*inner_cols, num_dimensions, CV_32FC1);
  cout << "innerrows: " << inner_rows << "\tinnercols: " << inner_cols << endl;
  int i = 0;
  //float* fm_row_ptr = feature_matrix.ptr<float>(i);

  vector<float> F;
  for (int r = h_size; r < (GPM[l].rows - h_size); r++)
  {

    for (int c = h_size; c < (GPM[l].cols - h_size); c++)
    {
      create_feature_vector(F, GPM, GPM_p, r, c, l, disp_default);
      assert(num_dimensions == F.size());
      //copy(F.begin(), F.end(), fm_row_ptr); 
      /*for (int j = 0; j < num_dimensions; j++)
      {
        (*fm_row_ptr) = F[j];
        fm_row_ptr++;
      }*/
      for (int j = 0; j < num_dimensions; j++)
      {
        feature_matrix.at<float>(i,j) = F[j];
      }
      i++;
      //if (i != inner_rows*inner_cols) fm_row_ptr = feature_matrix.ptr<float>(i);
    }

  }
  cout << "i = " << i << endl;
  cout << "innerrows*innercols = " << inner_rows*inner_cols << endl;
  
}

void ImageAnalogy::create_image_analogy(const Mat& A, const Mat& A_p, 
                                        const Mat& B, Mat& B_p)
{
  //test_flann();
  //return;
  //cout << "started image analogy creation" << endl;
  B_p = Mat::zeros(B.rows, B.cols, CV_8UC1);
  Mat s = Mat::zeros(B.rows, B.cols, CV_16UC2);
  vector<Mat> GPA, GPA_p, GPB, GPB_p, PS;
  construct_gaussian(A, GPA);
  construct_gaussian(A_p, GPA_p);
  construct_gaussian(B, GPB);
  construct_gaussian(B_p, GPB_p);

  construct_gaussian(s, PS);

  ImagePyramids IP;
  IP.GPA = GPA;
  IP.GPA_p = GPA_p;
  IP.GPB = GPB;
  IP.GPB_p = GPB_p;
  //construct source matrix vector
  //feature stuff
  cout << "n_size = " << n_size << " h_size = " << h_size << " L = " << L << endl; 
  //cout << "constructing gaussian pyramids" << endl;
  //FLANN stuff, create flann pyramid
  
  for (int l = L-1; l >= 0; l--)
  {
    cout << "level " << l << " of the alg" << endl;
    //for each point q in B_p
    int rows = IP.GPB_p[l].rows;
    int cols = IP.GPB_p[l].cols;

    flann::KDTreeIndexParams indexParams(4);
    //flann::LinearIndexParams indexParams; 
    Mat feature_matrix = Mat::zeros(1,1,CV_32FC1);
    cout << "gunna make feature_matrix" << endl;
    create_feature_matrix(l, IP.GPA, IP.GPA_p, feature_matrix);
    //cout << feature_matrix << endl;
    cout << "made feature_matrix" << endl;
    flann::Index kdtree(feature_matrix, indexParams);

    for (int r = 0; r < rows; r++)
    {
      cout << "row " << r << " of the alg" << endl;
      uchar* bp_row_ptr = IP.GPB_p[l].ptr<uchar>(r);
      Vec2s* source_row_ptr = PS[l].ptr<Vec2s>(r); 
      for ( int c = 0; c < cols; c++)
      {
        Point q = Point(c,r);
        //cout << "q = " << q << endl;
        Point p = best_match(IP, PS, l, q, kdtree, feature_matrix);
        //cout << "p = " << p << endl;
        //cout <<  endl;
        *bp_row_ptr = IP.GPA_p[l].at<uchar>(p.y, p.x);
        (*source_row_ptr)[0] = p.y;
        (*source_row_ptr)[1] = p.x;
        bp_row_ptr++;
        source_row_ptr++;
      }
    }
    //namedWindow("Bp_current", CV_WINDOW_AUTOSIZE);
    //imshow("Bp_current", IP.GPB_p[l]);
    //waitKey(0);
  }

  B_p = IP.GPB_p[0];
}

void ImageAnalogy::test_flann()
{
  int num_dimensions = 2;
  Mat feature_matrix;
  feature_matrix.create(3*3, num_dimensions, CV_32FC1);
  int i = 0;
  for (int r = 0; r < 3; r++)
  {
    for (int c = 0; c < 3; c++)
    {
      feature_matrix.at<float>(i,0) = r;
      feature_matrix.at<float>(i,1) = c;
      i++;
    }
  }

  vector<float> query;
  query.push_back(2); 
  query.push_back(1); 
  flann::KDTreeIndexParams indexParams(4);
  flann::Index kdtree(feature_matrix, indexParams);
  

  vector<int> index(1);
  vector<float> dist(1);

  kdtree.knnSearch(query, index, dist, 1, flann::SearchParams(128));
  
  Point p;
  p.x = (index[0] % (7 - 2*h_size)) + h_size;
  p.y = (index[0] / (7 - 2*h_size)) + h_size;

  cout << "P: " << p;

  cout << " index= " << index[0] << " dist = " << dist[0] << endl;
   

}
