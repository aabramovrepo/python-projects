// Alexey Abramov <abramov@physik3.gwdg.de>

#include "CColor.h"
#include <CSegmentation_cuda.h>
#include <dense_disparity.h>

#include <iostream>
#include <fstream>
#include <string.h>

#include <cutil.h>
#include <cutil_inline.h>

#include <CudaDenseVision.h>

#include <gpu/gpu.hpp>

#define SC(x) cutilSafeCall(x)
#define CE(x) cutilCheckError(x)

int pre_shift_value = 0;

void store_labels(CColor *pColors, unsigned int *ptr, std::string &filename, int h, int w);

/** Store values to the file (taking the data format into account)
  * @param str file name
  * @param pData pointer to the data that needs to be stored
  * @param w width
  * @param h height
  */
template <class T> void store_data(std::string& str, T *pData, int w, int h);

/** Estimate sparse disparity map by the phase-based stereo algorithm
  * @param pLeft left input image (OpenCV)
  * @param pRight right input image (OpenCV)
  * @param pDisp_incons pointer to the buffer with inconsistent disparity map
  * @param pDisp_cons pointer to the buffer with consistent disparity map
  * @param consistency_thr consistency threshold
  * @param width  width of the image
  * @param height  height of the image
  */
void ComputeDisparity(IplImage *pLeft, IplImage *pRight, float *pDisp_incons, float *pDisp_cons, float consistency_thr, int width, int height);

void SegmentationRefinement(unsigned char *pImage, unsigned int *pLabels, int w, int h);

void RunBenchmarksOpenCV();

/*------------------------------------------------------------------------------------------------------------------------------------*/
int main(int argc, char **argv){
  
  std::string fname_left = std::string(argv[1]);
  std::string fname_right = std::string(argv[2]);
  std::string output_path = std::string(argv[3]);
  float mul_t_left_base = atof(argv[4]);
  pre_shift_value = atoi(argv[5]);

  std::cout << "fname_left = " << fname_left << ", fname_right = " << fname_right << ", output_path = " << output_path << ", mul = " << mul_t_left_base << ", pre_shift_value = " << pre_shift_value << std::endl;
  
//  RunBenchmarksOpenCV();
//  return 0;//-->

  // set image size
  int h = 256;//1024;//512;//128;//256;
  int w = 320;//1280;//640;//160;//320;

  CColor *pColors = new CColor(1000);

  // <RGB left image> | <RGB right image>
  unsigned char *pInputData = new unsigned char[2*3*w*h];
  bzero(pInputData, 2*3*w*h*sizeof(unsigned char));

  float *pFlowX_left = new float[w * h];
  float *pFlowY_left = new float[w * h];

  bzero(pFlowX_left, h*w*sizeof(float));
  bzero(pFlowY_left, h*w*sizeof(float));

  float *pFlowX_right = new float[w * h];
  float *pFlowY_right = new float[w * h];

  bzero(pFlowX_right, h*w*sizeof(float));
  bzero(pFlowY_right, h*w*sizeof(float));

  // inconsistent disparity map
  float *pDisparity_incons = new float[w * h];
  bzero(pDisparity_incons, w * h * sizeof(float));

  // consistent disparity map
  float *pDisparity_cons = new float[w * h];
  bzero(pDisparity_cons, w * h * sizeof(float));

  // segmentation parameters
  int n_left_base = 10;
  int n_left_relax = 40;//10;
  int n_right_relax = 40;

  //float mul_t_left_base = 0.92f;//0.9f;//0.75f;//0.75f;//1.0f;//0.9f;//1.0f;//1.0f;
  float mul_t_left_relax = 5.0f;//4.0f;
  float mul_t_right_relax = 5.0f;//4.0f;

  float T1 = 0.5f;
  float T2 = 0.5f;
  float mulSA = 0.999f;//1.0f;

  SegmentTracking::SegmentationParams _data(h, w, n_left_base, n_left_relax, n_right_relax, T1, T2, mulSA,mul_t_left_base, mul_t_left_relax,
                                            mul_t_right_relax, SegmentTracking::cStereoStream, SegmentTracking::cCIELABColorSpace, SegmentTracking::cNoRefinement, 0);

  // read input data
  IplImage *framePtrLeft, *framePtrRight, *framePtrLeft_smoothed, *framePtrRight_smoothed;

  IplImage *pLeftSrc = cvLoadImage(fname_left.c_str());
  IplImage *pRightSrc = cvLoadImage(fname_right.c_str());

  if(!pLeftSrc || !pRightSrc){
    std::cerr << "Cannot not load image files !" << std::endl;
    return 0;//-->
  }

  framePtrLeft = cvCreateImage(cvSize(w,h), pLeftSrc->depth, pLeftSrc->nChannels);
  cvResize(pLeftSrc, framePtrLeft);

  framePtrRight = cvCreateImage(cvSize(w,h), pRightSrc->depth, pRightSrc->nChannels);
  cvResize(pRightSrc, framePtrRight);

  framePtrLeft_smoothed = cvCreateImage(cvGetSize(framePtrLeft), framePtrLeft->depth, framePtrLeft->nChannels);
  cvSmooth( framePtrLeft, framePtrLeft_smoothed, CV_GAUSSIAN, 3, 3 );

  framePtrRight_smoothed = cvCreateImage(cvGetSize(framePtrRight), framePtrRight->depth, framePtrRight->nChannels);
  cvSmooth( framePtrRight, framePtrRight_smoothed, CV_GAUSSIAN, 3, 3 );

  // save input images
  //std::string fname = std::string("./img-left-input.png");
  std::string fname = output_path + std::string("img-left-input.png");
  cvSaveImage(fname.c_str(), framePtrLeft);

  //fname = std::string("./img-right-input.png");
  fname = output_path + std::string("img-right-input.png");
  cvSaveImage(fname.c_str(), framePtrRight);

  // save input images after smoothing
  //fname = std::string("./img-left-smooth.png");
  //cvSaveImage(fname.c_str(), framePtrLeft_smoothed);

  //fname = std::string("./img-right-smooth.png");
  //cvSaveImage(fname.c_str(), framePtrRight_smoothed);

  // get consistent and inconsistent disparity maps
  float consist_threshold = 0.4f;//0.8f;
  ComputeDisparity(framePtrLeft, framePtrRight, pDisparity_incons, pDisparity_cons, consist_threshold, w, h);

  // define min and max disparity values and replace 'NaN's by zeros (?)
  float max_disp_val = 0.0f;
  float min_disp_val = 0.0f;

  for(int ind = 0; ind < w*h; ++ind){

//    if( isnan( *(pDisparity_incons + ind) ) )
//      *(pDisparity_incons + ind) = 0;
//    else
//      *(pDisparity_incons + ind) = abs( *(pDisparity_incons + ind) );

//    if( isnan( *(pDisparity_cons + ind) ) )
//      *(pDisparity_cons + ind) = 0;
//    else{

      if( min_disp_val > *(pDisparity_incons + ind) )
        min_disp_val = *(pDisparity_incons + ind);

      if( max_disp_val < *(pDisparity_incons + ind) )
        max_disp_val = *(pDisparity_incons + ind);

//      *(pDisparity_cons + ind) = abs( *(pDisparity_cons + ind) );

//    }
  }

  std::cout << "min_disp_val = " << min_disp_val << std::endl;
  std::cout << "max_disp_val = " << max_disp_val << std::endl;
  
  // save inconsistent disparity map to file
  //fname = std::string("./sparse_disparity_incons.dat");
  fname = output_path + std::string("sparse_disparity_incons.dat");
  store_data<float>(fname, pDisparity_incons, w, h);

  // save consistent disparity map to file
  //fname = std::string("./sparse_disparity_cons.dat");
  fname = output_path + std::string("sparse_disparity_cons.dat");
  store_data<float>(fname, pDisparity_cons, w, h);


  // separate R,G,B components creating data structures like: R_left|R_right, G_left|G_right, B_left|B_right
  for(int i = 0; i < h; ++i){
    for(int j = 0; j < w; ++j){
/*
      // the first image: |R|G|B|
      *(pInputData + i*w + j) = *(framePtrLeft->imageData + i*framePtrLeft->widthStep + j*framePtrLeft->nChannels + 2);
      *(pInputData + h*w + i*w + j) = *(framePtrLeft->imageData + i*framePtrLeft->widthStep + j*framePtrLeft->nChannels + 1);
      *(pInputData + 2*h*w + i*w + j) = *(framePtrLeft->imageData + i*framePtrLeft->widthStep + j*framePtrLeft->nChannels);

      // the second image: |R|G|B|
      *(pInputData + 3*w*h + i*w + j) = *(framePtrRight->imageData + i*framePtrRight->widthStep + j*framePtrRight->nChannels + 2);
      *(pInputData + 3*w*h + h*w + i*w + j) = *(framePtrRight->imageData + i*framePtrRight->widthStep + j*framePtrRight->nChannels + 1);
      *(pInputData + 3*w*h + 2*h*w + i*w + j) = *(framePtrRight->imageData + i*framePtrRight->widthStep + j*framePtrRight->nChannels);
*/

      // the first image: |R|G|B|
      *(pInputData + i*w + j) = *(framePtrLeft_smoothed->imageData + i*framePtrLeft_smoothed->widthStep + j*framePtrLeft_smoothed->nChannels + 2);
      *(pInputData + h*w + i*w + j) = *(framePtrLeft_smoothed->imageData + i*framePtrLeft_smoothed->widthStep + j*framePtrLeft_smoothed->nChannels + 1);
      *(pInputData + 2*h*w + i*w + j) = *(framePtrLeft_smoothed->imageData + i*framePtrLeft_smoothed->widthStep + j*framePtrLeft_smoothed->nChannels);

      // the second image: |R|G|B|
      *(pInputData + 3*w*h + i*w + j) = *(framePtrRight_smoothed->imageData + i*framePtrRight_smoothed->widthStep + j*framePtrRight_smoothed->nChannels + 2);
      *(pInputData + 3*w*h + h*w + i*w + j) = *(framePtrRight_smoothed->imageData + i*framePtrRight_smoothed->widthStep + j*framePtrRight_smoothed->nChannels + 1);
      *(pInputData + 3*w*h + 2*h*w + i*w + j) = *(framePtrRight_smoothed->imageData + i*framePtrRight_smoothed->widthStep + j*framePtrRight_smoothed->nChannels);
    }
  }

  // create segmentation object
  SegmentTracking::CSegmentation *pSegmentation = new SegmentTracking::CSegmentation(_data);

  // create stereo vision object
  DenseDisparity::CStereoVision *pStereoVision = new DenseDisparity::CStereoVision(h,w);

  // set frames for segmentation
  cv::Mat SensoryData = cv::Mat::zeros(h,w,CV_16UC1);
  const short segment_size = 10;//50;//50;//100;
  pSegmentation->SetFrames(pInputData, (pInputData + 3*w*h), SensoryData, segment_size, false);

  // segment stereo pair (no optical flow is used here actually)
  //pSegmentation->SegmentStereoPair(SegmentTracking::cSegmentationBase, pFlowX, pFlowY, pDisparity_incons/*, SensoryData*/);
  pSegmentation->SegmentStereoPair(SegmentTracking::cSegmentationBase, pFlowX_left, pFlowY_left, pFlowX_right, pFlowY_right, pDisparity_incons);
  
  unsigned int *pLeft = 0;
  unsigned int *pRightInit = 0;
  unsigned int *pRight = 0;
  
  // save left and right segments as images
  pLeft = pSegmentation->GetSpinVariables(0);
  pRightInit = pSegmentation->GetBuffer();
  pRight = pSegmentation->GetSpinVariables(1);

  // prepare stereo segments for dense disparity estimation
  pStereoVision->SetFrames(pLeft, pRight, pDisparity_cons);
  
  pLeft = pStereoVision->GetLeftSegments();
  pRight = pStereoVision->GetRightSegments();

  //pLeft = pSegmentation->GetSpinVariables(0);
  //pRight = pSegmentation->GetSpinVariables(1);

  // do the segmentation refinement
  //SegmentationRefinement(pInputData, pLeft, w, h);

  // save left and right labels
  //fname = std::string("./segments-left.dat");
  fname = output_path + std::string("segments-left.dat");
  store_data<unsigned int>(fname, pLeft, w, h);

  //fname = std::string("./segments-right.dat");
  fname = output_path + std::string("segments-right.dat");
  store_data<unsigned int>(fname, pRight, w, h);

  // save left and right segments as images
  //std::string fname_lb_left = std::string("./segments-left.png");
  std::string fname_lb_left = output_path + std::string("segments-left.png");
  //std::string fname_lb_right = std::string("./segments-right.png");
  std::string fname_lb_right = output_path + std::string("segments-right.png");

  store_labels(pColors, pLeft, fname_lb_left, h, w);
  store_labels(pColors, pRight, fname_lb_right, h, w);

  // run dense disparity estimation
  const short optimization_type = DenseDisparity::cLeastSquares;
  //const short optimization_type = DenseDisparity::cNelderMead;
  const short ndim = 3;//5;//3;
  const short edge_disp_shift = 5;
  const short edge_disp_max = 50;
  const short avg_disp_type = DenseDisparity::cAverageSparseEdgeDisp;
  const float sig_edge = 0.3;
  const float sig_object = 1.0;
  const short occl_max_depth = 3;//1;//1;
  const short occlusion_size = 10;
  const short vertical_step = 2;//5;
  const short range_value = 3;//2;
  
  double time = 0.0;
  unsigned int timerGPU = 0;
  CE( cutCreateTimer(&timerGPU) );

  CE( cutResetTimer(timerGPU) );
  CE( cutStartTimer(timerGPU) );

  pStereoVision->EstimateDenseDisparityMap(optimization_type, ndim, edge_disp_shift, edge_disp_max, avg_disp_type, sig_edge, sig_object,
                                           occl_max_depth, occlusion_size, vertical_step, range_value);

  CE( cutStopTimer(timerGPU) );
  time = cutGetTimerValue(timerGPU);
  std::cout << "---> Dense disparity estimation, t = " << time << " ms" << std::endl;

  float *pAverageDisparity = pStereoVision->GetAverageDisparityMap();
  float *pAverageDisparity_lines = pStereoVision->GetAverageDisparityLinesMap();
  float *pEdgeDisparity = pStereoVision->GetEdgeDisparityMap();
  float *pFinalDisparity = pStereoVision->GetDenseDisparityMap();
  float *pInitialDisparity = pStereoVision->GetInitialDisparityMap();
  float *pSigma = pStereoVision->GetSigmaMap();
  bool *pOcclusionMap = pStereoVision->GetOcclusionMap();
  bool *pEdgeDisparityMask = pStereoVision->GetEdgeDisparityMask();
  bool *pOutliersMap = pStereoVision->GetOutliersMap();

  float *pSparseDisparity = pStereoVision->GetSparseDisparityMap();

  // replace zeros by NaN's for better plots in python
  for(int ind = 0; ind < h*w; ++ind){

    if( !*(pAverageDisparity + ind) )
      *(pAverageDisparity + ind) = NAN;

    if( !*(pAverageDisparity_lines + ind) )
      *(pAverageDisparity_lines + ind) = NAN;

    if( !*(pInitialDisparity + ind) )
      *(pInitialDisparity + ind) = NAN;

//    if( !*(pEdgeDisparity + ind) )
//      *(pEdgeDisparity + ind) = NAN;

    if( !*(pFinalDisparity + ind) )
      *(pFinalDisparity + ind) = NAN;

    if( !*(pSigma + ind) )
      *(pSigma + ind) = NAN;

    if( !*(pSparseDisparity + ind) )
      *(pSparseDisparity + ind) = NAN;

//    if( *(pOcclusionMap + ind) == false )
//      *(pOcclusionMap + ind) = NAN;

  }

  // save filtered sparse disparity map to file
  //fname = std::string("./sparse-disparity-filtered.dat");
  fname = output_path + ::string("sparse-disparity-filtered.dat");
  store_data<float>(fname, pSparseDisparity, w, h);

  // save average disparity map to file
  //fname = std::string("./average-disparity.dat");
  fname = output_path + std::string("average-disparity.dat");
  store_data<float>(fname, pAverageDisparity, w, h);

  // save average disparity map (lines) to file
  //fname = std::string("./average-disparity-lines.dat");
  fname = output_path + std::string("average-disparity-lines.dat");
  store_data<float>(fname, pAverageDisparity_lines, w, h);

  // save edge disparity map to file
  //fname = std::string("./edge-disparity.dat");
  fname = output_path + std::string("edge-disparity.dat");
  store_data<float>(fname, pEdgeDisparity, w, h);

  // save final disparity map to file
  //fname = std::string("./final-disparity.dat");
  //store_data<float>(fname, pFinalDisparity, w, h);

  // save initial disparity map to file
  //fname = std::string("./initial-disparity.dat");
  fname = output_path + std::string("initial-disparity.dat");
  store_data<float>(fname, pInitialDisparity, w, h);

  // save sigma map to file
  //fname = std::string("./sigma-map.dat");
  fname = output_path + std::string("sigma-map.dat");
  store_data<float>(fname, pSigma, w, h);

  // save occlusion map to file
  //fname = std::string("./occlusion-map.dat");
  fname = output_path + std::string("occlusion-map.dat");
  store_data<bool>(fname, pOcclusionMap, w, h);

  // save edge disparity mask to file
  //fname = std::string("./edge-disparity-mask.dat");
  fname = output_path + std::string("edge-disparity-mask.dat");
  store_data<bool>(fname, pEdgeDisparityMask, w, h);

  //fname = std::string("./outliers-map.dat");
  fname = output_path + std::string("outliers-map.dat");
  store_data<bool>(fname, pOutliersMap, w, h);

  for(int ind = 0; ind < h*w; ++ind){

    if( *(pEdgeDisparity + ind) && *(pEdgeDisparityMask + ind) )
      *(pEdgeDisparity + ind) = 0;

  }

  // save final edge disparity map to file
  //fname = std::string("./edge-disparity-final.dat");
  fname = output_path + std::string("edge-disparity-final.dat");
  store_data<float>(fname, pEdgeDisparity, w, h);

  delete[] pInputData;
  delete[] pFlowX_left;
  delete[] pFlowY_left;
  delete[] pFlowX_right;
  delete[] pFlowY_right;
  delete[] pDisparity_cons;
  delete[] pDisparity_incons;

  delete pSegmentation;
  delete pStereoVision;
  delete pColors;

  return 0;
}
/*------------------------------------------------------------------------------------------------------------------------------------*/
void SegmentationRefinement(unsigned char *pImage, unsigned int *pLabels, int w, int h){
  
  int n_n = 8;
  int shift_x[] = {-1,  0,  1, -1, 1, -1, 0, 1};
  int shift_y[] = {-1, -1, -1,  0, 0,  1, 1, 1};

  // do segmentation refinement
  for(int i = 0; i < h; ++i){
    for(int j = 0; j < w; ++j){

      //int cnt = 0;
      unsigned int lb = *(pLabels + i*w + j);

      if(lb)
        continue;//--^

      float color_diff = 500.0f;

      for(int ind = 0; ind < n_n; ++ind){

        int c_y = i + shift_y[ind];
        int c_x = j + shift_x[ind];

        if(c_y < 0 || c_y >= h || c_x < 0 || c_x >= w)
          continue;//--^

        unsigned int neighb_lb = *(pLabels + c_y*w + c_x);

        if(!neighb_lb)
          continue;//--^

        float color_dst = 0.0;

        float tmp = *(pImage + i*w + j) - *(pImage + c_y*w + c_x);
        color_dst += tmp*tmp;

        tmp = *(pImage + h*w + i*w + j) - *(pImage + h*w + c_y*w + c_x);
        color_dst += tmp*tmp;

        tmp = *(pImage + 2*h*w + i*w + j) - *(pImage + 2*h*w + c_y*w + c_x);
        color_dst += tmp*tmp;

        color_dst = sqrtf(color_dst);

        if( color_dst < color_diff){

          color_diff = color_dst;
          *(pLabels + i*w + j) = neighb_lb;
        }

      }
    }
  }

}
/*------------------------------------------------------------------------------------------------------------------------------------*/
void RunBenchmarksOpenCV(){

  int gpu_cnt = cv::gpu::getCudaEnabledDeviceCount();
  std::cout << "gpu_cnt = " << gpu_cnt << std::endl;

//  cv::Mat img_l = cv::imread("./dataset/Justus/red-plate/left.png", 0);
//  cv::Mat img_r = cv::imread("./dataset/Justus/red-plate/right.png", 0);

  cv::Mat img_l = cv::imread("./dataset/Lab/Box-2/left.png", 0);
  cv::Mat img_r = cv::imread("./dataset/Lab/Box-2/right.png", 0);

  int height = img_l.size().height;
  int width = img_l.size().width;

  std::cout << "height = " << height << ", width = " << width << std::endl;

  cv::gpu::GpuMat left_gpu, right_gpu;
  cv::gpu::GpuMat disp_bm, disp_bp, disp_csbp;


  // Block matching (BM)
  //cv::gpu::StereoBM_GPU bm(0, 128, 19);
  cv::gpu::StereoBM_GPU bm(0);

  // Belief propagation (BP)
  cv::gpu::StereoBeliefPropagation bpm(40, 20, 4, 100, 0.1f, 500, 1, CV_32F);

  // Constant space belief propagation (CSBP)
  cv::gpu::StereoConstantSpaceBP csbp(128, 16, 4, 4);

  left_gpu.upload(img_l);
  right_gpu.upload(img_r);

  bm(left_gpu, right_gpu, disp_bm);
  bpm(left_gpu, right_gpu, disp_bp);
  csbp(left_gpu, right_gpu, disp_csbp);

  cv::Mat disparity_bm, disparity_bp, disparity_csbp;

  disp_bm.convertTo(disp_bm, CV_16S);
  disp_bp.convertTo(disp_bp, CV_16S);
  disp_csbp.convertTo(disp_csbp, CV_16S);

  disp_bm.download(disparity_bm);
  disp_bp.download(disparity_bp);
  disp_csbp.download(disparity_csbp);

  disp_bm.convertTo(disp_bm, CV_8U);
  disp_bp.convertTo(disp_bp, CV_8U);
  disp_csbp.convertTo(disp_csbp, CV_8U);

  cv::imwrite("./left.png", img_l);
  cv::imwrite("./right.png", img_r);
//  cv::imwrite("./disparity.png", disp);

  // save obtained disparity maps
  std::ofstream fil_bm, fil_bp, fil_csbp;

  std::string fname_bm = "./disparity_bm.dat";
  std::string fname_bp = "./disparity_bp.dat";
  std::string fname_csbp = "./disparity_csbp.dat";

  fil_bm.open(fname_bm.c_str());
  fil_bp.open(fname_bp.c_str());
  fil_csbp.open(fname_csbp.c_str());

  if(!fil_bm || !fil_bp || !fil_csbp){
    std::cerr << "Unable to open file" << std::endl;
    return;//-->
  }

  int cnt = 0;

  for(int i = 0; i < height; ++i){
    for(int j = 0; j < width; ++j){

      fil_bm << disparity_bm.at<short>(i,j) << " ";
      fil_bp << disparity_bp.at<short>(i,j) << " ";
      fil_csbp << disparity_csbp.at<short>(i,j) << " ";
      cnt++;

      if(cnt == width){
        fil_bm << std::endl;
        fil_bp << std::endl;
        fil_csbp << std::endl;
        cnt = 0;
      }

    }
  }

  fil_bm.close();
  fil_bp.close();
  fil_csbp.close();

}
/*------------------------------------------------------------------------------------------------------------------------------------*/
void ComputeDisparity(IplImage *pLeft, IplImage *pRight, float *pDisp_incons, float *pDisp_cons, float consistency_thr, int width, int height){

  // compute disparity maps: consistent and inconsistent
  CvSize frameSize;
  frameSize = cvGetSize(pLeft);

  IplImage *auxBW = cvCreateImage(frameSize,IPL_DEPTH_8U,1);
  IplImage *curr_frameLeft = cvCreateImage(frameSize,IPL_DEPTH_32F,1);
  IplImage *curr_frameRight = cvCreateImage(frameSize,IPL_DEPTH_32F,1);

  cvConvertImage(pLeft, auxBW);
  cvConvertScale(auxBW, curr_frameLeft, 1.0);

  cvConvertImage(pRight, auxBW);
  cvConvertScale(auxBW, curr_frameRight, 1.0);

  // Setup Stereo Engine
  float *d_leftPixels, *d_rightPixels;
  size_t d_pixelsPitch;
  cudaMallocPitch((void **)&d_leftPixels, &d_pixelsPitch, frameSize.width*sizeof(float), frameSize.height);
  cudaMallocPitch((void **)&d_rightPixels, &d_pixelsPitch, frameSize.width*sizeof(float), frameSize.height);

  int h_pixelPitch = frameSize.width*sizeof(float);
  cudaMemcpy2D(d_leftPixels, d_pixelsPitch, curr_frameLeft->imageData, h_pixelPitch, frameSize.width*sizeof(float), frameSize.height, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_rightPixels, d_pixelsPitch, curr_frameRight->imageData, h_pixelPitch, frameSize.width*sizeof(float), frameSize.height, cudaMemcpyHostToDevice);

  D_Stereo *d_stereo = new D_Stereo(d_leftPixels, d_rightPixels, d_pixelsPitch, frameSize.width, frameSize.height, 5, true, true, 1.0f, 0);
  float *h_stereo = (float *)malloc(frameSize.width*frameSize.height*sizeof(float));

  float lowerLim, upperLim;
  int lowerLim_1000 = 0;
  int upperLim_1000 = 1000;
  int pre_shift = pre_shift_value;
  int consistent = 0;//1;//0;
  int median_filter = 1;//0;//1;

  // Start Processing
  CvPoint textCorner = cvPoint(20, 20);
  CvSize infoImageSize = cvSize(300, 200);

  IplImage *stereo = cvCreateImage(frameSize, IPL_DEPTH_32F, 1);

  lowerLim = (float)(lowerLim_1000-500)/10.0f;
  upperLim = (float)(upperLim_1000-500)/10.0f;

  //d_stereo->setParameters((bool)median_filter, (bool)consistent, 0.7f/*1.0f*//*0.9f*/, pre_shift-50);
  d_stereo->setParameters((bool)median_filter, (bool)consistent, 0.7f/*1.0f*//*0.9f*/, pre_shift-50);

  cvConvertImage(pLeft, auxBW);
  cvConvertScale(auxBW, curr_frameLeft, 1.0);
  cvConvertImage(pRight, auxBW);
  cvConvertScale(auxBW, curr_frameRight, 1.0);

  // Update gabor pyramids
  cudaThreadSynchronize();

  cudaMemcpy2D(d_leftPixels, d_pixelsPitch, curr_frameLeft->imageData, h_pixelPitch, frameSize.width*sizeof(float), frameSize.height, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_rightPixels, d_pixelsPitch, curr_frameRight->imageData, h_pixelPitch, frameSize.width*sizeof(float), frameSize.height, cudaMemcpyHostToDevice);

  d_stereo->updateImages(d_leftPixels, d_rightPixels);
  cudaThreadSynchronize();

  d_stereo->update();
  d_stereo->getStereo((float *)stereo->imageData);
  cudaThreadSynchronize();

  memcpy(pDisp_incons, (float *)stereo->imageData, frameSize.width*frameSize.height*sizeof(float));

  consistent = 1;
  d_stereo->setParameters((bool)median_filter, (bool)consistent, consistency_thr, pre_shift-50);
  cudaThreadSynchronize();

  d_stereo->updateImages(d_leftPixels, d_rightPixels);
  cudaThreadSynchronize();

  d_stereo->update();
  d_stereo->getStereo((float *)stereo->imageData);
  cudaThreadSynchronize();

  memcpy(pDisp_cons, (float *)stereo->imageData, frameSize.width*frameSize.height*sizeof(float));

  delete(d_stereo);
  cudaFree(d_leftPixels);
  cudaFree(d_rightPixels);
  free(h_stereo);

}
/*------------------------------------------------------------------------------------------------------------------------------------*/
void store_labels(CColor *pColors, unsigned int *ptr, std::string &filename, int h, int w){

  int channels = 3;
  int depth = 8;

  IplImage *img = cvCreateImage(cvSize(w,h),depth,channels);

  for(int i = 0; i < h; ++i){
    for(int j = 0; j < w; ++j){

      int val = *(ptr + i*w + j);
      
      colorRGB clr;
      pColors->GetColor(val, clr);

      if( clr.m_R == 255 && clr.m_G == 255 && clr.m_B == 255 ){

        *(img->imageData + i*img->widthStep + j*channels + 2) = 255;
        *(img->imageData + i*img->widthStep + j*channels + 1) = 0;
        *(img->imageData + i*img->widthStep + j*channels ) = 0;
      }
      else{

        *(img->imageData + i*img->widthStep + j*channels + 2) = clr.m_R;
        *(img->imageData + i*img->widthStep + j*channels + 1) = clr.m_G;
        *(img->imageData + i*img->widthStep + j*channels ) = clr.m_B;
      }
    }
  }

  cvSaveImage(filename.c_str(),img);
  cvReleaseImage(&img);
}
/*------------------------------------------------------------------------------------------------------------------------------------*/
template <class T> void store_data(std::string& str, T *pData, int w, int h){

  std::ofstream f;
  f.open(str.c_str());

  if( !f )
    std::cerr << "Unable to open file" << std::endl;

  int cnt = 0;

  for(int i = 0; i < w*h; ++i){

    f << *(pData + i) << " ";

    cnt++;

    if(cnt == w){
      f << std::endl;
      cnt = 0;
    }

  }

  f.close();

}
/*------------------------------------------------------------------------------------------------------------------------------------*/
