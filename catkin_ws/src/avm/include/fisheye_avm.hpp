#pragma once
#ifndef FISHEYE_AVM_HPP

// #define USE_CUDA

#include <Eigen/Core>
#include <opencv2/core.hpp>

#ifdef USE_CUDA
    #include <opencv2/core/cuda.hpp>
#endif



class FisheyeAVM {
 public:
  FisheyeAVM() = default;

  void open(const std::vector<std::string> &filenames);

  bool initialize();

  void operator()(const std::vector<cv::Mat> &imgs, cv::Mat &result) const;

  cv::Mat operator()(const std::vector<cv::Mat> &imgs) const;

  void set_output_resolution(const cv::Size2i &s) {
    output_width = s.width;
    output_height = s.height;
  }

  cv::Size2i get_output_resolution() const {
    return {output_width, output_height};
  }

  void set_scale(double s) { output_scale = s; }

  double get_scale() const { return output_scale; }

  void set_interpolation_mode(int mode) { cv_interpolation_mode = mode; }

  int get_interpolation_mode() const { return cv_interpolation_mode; }

  void set_logging(bool enable) { logging = enable; }

  bool get_logging() const { return logging; }

 private:
  // parameters
  int output_width = 0;     // unit: [pixel]
  int output_height = 0;    // unit: [pixel]
  double output_scale = 0;  // unit: [mm/pixel]
  int cv_interpolation_mode =
      0;  // 0: nearest, 1: bilinear ... detail cv::INTER_XXX

  // logging mode
  bool logging = false;

  //
  int frames = 0;

  // load from yaml files
  std::vector<int> widths;
  std::vector<int> heights;
  std::vector<Eigen::Matrix2d> stretchs;
  std::vector<Eigen::Vector4d> coeffs;
  std::vector<Eigen::Vector2d> centers;
  std::vector<Eigen::Matrix3d> transforms;

  // set during initialization
  std::vector<std::vector<double>> interpolations;
  std::vector<cv::Mat> masks;
  std::vector<cv::Mat> mapxs;
  std::vector<cv::Mat> mapys;
  std::vector<cv::Rect> rois;
  std::vector<cv::Mat> warp8us;
  std::vector<cv::Mat> warp32fs;
  std::vector<cv::Mat> img_tems;

  // host buffer
  mutable cv::Mat canvas;
  #ifdef USE_CUDA
   // cuda buffer
   mutable std::vector<cv::cuda::GpuMat> cu_masks;
   mutable std::vector<cv::cuda::GpuMat> cu_mapxs;
   mutable std::vector<cv::cuda::GpuMat> cu_mapys;
   mutable std::vector<cv::cuda::GpuMat> cu_imgs;
   mutable std::vector<cv::cuda::GpuMat> cu_warp8us;
   mutable std::vector<cv::cuda::GpuMat> cu_warp32fs;
   mutable cv::cuda::GpuMat cu_canvas;
   // cuda workflow
   mutable cv::cuda::Stream stream;
   mutable std::vector<cv::cuda::Stream> stream_forks;
   mutable cv::cuda::Event event;
   mutable std::vector<cv::cuda::Event> event_forks;
  #endif
};

#endif /* FISHEYE_AVM_HPP */