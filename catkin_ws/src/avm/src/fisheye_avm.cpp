
#include "fisheye_avm.hpp"

#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>

#include <opencv2/core.hpp>

#include <opencv2/core/eigen.hpp>

#ifdef USE_CUDA
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/core/cuda.hpp>
#endif
using namespace std::chrono;

#define REF(var, idx) auto &var = var##s[idx]

#define TICK(tag) auto tag##1 = std::chrono::steady_clock::now()

#define TOCK(tag)                                                           \
  auto tag##2 = std::chrono::steady_clock::now();                           \
  do {                                                                      \
    if (!logging) break;                                                    \
    double ms = duration_cast<microseconds>(tag##2 - tag##1).count() / 1e3; \
    std::cout << "[" << #tag << "]: " << ms << "ms" << std::endl;           \
  } while (0)

#define LOG(stream)                   \
  do {                                \
    if (!logging) break;              \
    std::cout << stream << std::endl; \
  } while (0)

void FisheyeAVM::open(const std::vector<std::string> &filenames) {
  frames = filenames.size();
  widths.resize(frames);
  heights.resize(frames);
  stretchs.resize(frames);
  coeffs.resize(frames);
  centers.resize(frames);
  transforms.resize(frames);
  interpolations.resize(frames);
  masks.resize(frames);
  mapxs.resize(frames);
  mapys.resize(frames);
  rois.resize(frames);

  warp8us.resize(frames); //add
  warp32fs.resize(frames); //add
  img_tems.resize(frames);//add

  #ifdef USE_CUDA
   cu_masks.resize(frames);
   cu_mapxs.resize(frames);
   cu_mapys.resize(frames);
   cu_imgs.resize(frames);
   cu_warp8us.resize(frames);
   cu_warp32fs.resize(frames);
   stream_forks.resize(frames);
   event_forks.resize(frames);
  #endif

  for (size_t i = 0; i < frames; i++) {
    cv::FileStorage fs(filenames[i], cv::FileStorage::READ);
    fs["Width"] >> widths[i];
    fs["Height"] >> heights[i];
    cv::Mat stretch_cv = fs["StretchMatrix"].mat();
    cv::Mat coeff_cv = fs["MappingCoefficients"].mat();
    cv::Mat center_cv = fs["DistortionCenter"].mat();
    cv::Mat transform_cv = fs["TransformMatrix"].mat();
    cv::cv2eigen(stretch_cv, stretchs[i]);
    cv::cv2eigen(coeff_cv, coeffs[i]);
    cv::cv2eigen(center_cv, centers[i]);
    cv::cv2eigen(transform_cv, transforms[i]);
  }
}

bool FisheyeAVM::initialize() {
  if (output_width <= 0 || output_height <= 0 || output_scale <= 0)
    return false;
  if (frames <= 0) return false;

  TICK(initialize);

  LOG("initialize mapx mapy mask");
  for (int i = 0; i < frames; i++) {
    REF(width, i);
    REF(height, i);
    REF(stretch, i);
    REF(coeff, i);
    REF(center, i);
    REF(transform, i);
    REF(interpolation, i);
    REF(mask, i);
    REF(mapx, i);
    REF(mapy, i);
    REF(roi, i);
    roi.x = output_width;
    roi.y = output_height;
    roi.width = -output_width;
    roi.height = -output_height;

    interpolation.reserve(output_width + output_height);
    for (size_t j = 0; j < interpolation.capacity(); j++) {
      double rho = j;
      double rho2 = rho * rho;
      double rho3 = rho * rho2;
      double rho4 = rho * rho3;
      double alpha =
          coeff[0] + coeff[1] * rho2 + coeff[2] * rho3 + coeff[3] * rho4;
      if (alpha < 0) break;
      double beta = rho / alpha * coeff[0];
      interpolation.emplace_back(beta);
    }

    mask.create(cv::Size2i{output_width, output_height}, CV_32FC3);
    mapx.create(cv::Size2i{output_width, output_height}, CV_32FC1);
    mapy.create(cv::Size2i{output_width, output_height}, CV_32FC1);

    for (int j = 0; j < output_width * output_height; j++) {
      int x = j % output_width;
      int y = j / output_height;
      struct Empty {
      } empty;
      try {
        Eigen::Vector3d pt = {(x - output_width / 2.0) * output_scale,
                              (y - output_height / 2.0) * output_scale, 1.0};
        pt = transform * pt;
        if (pt.z() <= 0) throw empty;
        pt /= pt.z();
        Eigen::Vector2d pu = pt.head<2>() - center;
        double beta = pu.norm();
        int rho2 =
            std::upper_bound(interpolation.begin(), interpolation.end(), beta) -
            interpolation.begin();
        if (rho2 >= interpolation.size()) throw empty;
        int rho1 = rho2 - 1;
        if (rho1 < 0) throw empty;
        double beta1 = interpolation[rho1];
        double beta2 = interpolation[rho2];
        double rho = rho1 + (rho2 - rho1) / (beta2 - beta1) * (beta - beta1);
        Eigen::Vector2d po = stretch * (pu / beta * rho) + center;
        if (!(0 <= po.x() && po.x() <= width - 1)) throw empty;
        if (!(0 <= po.y() && po.y() <= height - 1)) throw empty;

        float w = 1.0 / (beta2 - beta1);
        mask.at<cv::Vec3f>(y, x) = {w, w, w};
        mapx.at<float>(y, x) = po.x();
        mapy.at<float>(y, x) = po.y();
        if (x < roi.x) {
          roi.width = roi.x + roi.width - x;
          roi.x = x;
        }
        if (y < roi.y) {
          roi.height = roi.y + roi.height - y;
          roi.y = y;
        }
        if (x > roi.x + roi.width) {
          roi.width = x - roi.x;
        }
        if (y > roi.y + roi.height) {
          roi.height = y - roi.y;
        }
      } catch (const Empty &) {
        mask.at<cv::Vec3f>(y, x) = {0, 0, 0};
        mapx.at<float>(y, x) = -1;
        mapy.at<float>(y, x) = -1;
      }
    }
  }

  LOG("normalize mask");
  for (int i = 0; i < output_width * output_height; i++) {
    int x = i % output_width;
    int y = i / output_height;
    float mask_sum = 0;
    for (int j = 0; j < frames; j++) {
      mask_sum += masks[j].at<cv::Vec3f>(y, x)[0];
    }
    if (mask_sum == 0) continue;
    for (int j = 0; j < frames; j++) {
      masks[j].at<cv::Vec3f>(y, x) /= mask_sum;
    }
  }

  LOG("initialize cuda buffer");
  for (int i = 0; i < frames; i++) {
    REF(mask, i);
    REF(mapx, i);
    REF(mapy, i);
    REF(roi, i);
    REF(warp8u, i);
    REF(warp32f, i);

    mask = mask(roi).clone();
    mapx = mapx(roi).clone();
    mapy = mapy(roi).clone();
    warp8u.create(mask.size(), CV_8UC3);
    warp32f.create(mask.size(), CV_32FC3);
    REF(width, i);
    REF(height, i);

    #ifdef USE_CUDA
     REF(cu_mask, i);
     REF(cu_mapx, i);
     REF(cu_mapy, i);
     REF(cu_img, i);
     REF(cu_warp8u, i);
     REF(cu_warp32f, i);

     cu_mask.upload(mask);
     cu_mapx.upload(mapx);
     cu_mapy.upload(mapy);
     cu_img.create(cv::Size2i{width, height}, CV_8UC3);
     cu_warp8u.create(cu_mask.size(), CV_8UC3);
     cu_warp32f.create(cu_mask.size(), CV_32FC3);
    #endif
  };
  canvas.create(cv::Size2i{output_width, output_height}, CV_8UC3); //add
  #ifdef USE_CUDA
   cu_canvas.create(cv::Size2i{output_width, output_height}, CV_8UC3);
  #endif


  TOCK(initialize);

  return true;
}

#ifdef USE_CUDA
void FisheyeAVM::operator()(const std::vector<cv::Mat> &imgs,
                            cv::Mat &result) const {
  TICK(operator);

  cu_canvas.setTo(0, stream);
  event.record(stream);
  for (int i = 0; i < frames; i++) {
    REF(stream_fork, i);
    REF(event_fork, i);
    REF(mask, i);
    REF(mapx, i);
    REF(mapy, i);
    REF(cu_mask, i);
    REF(cu_mapx, i);
    REF(cu_mapy, i);
    REF(cu_img, i);
    REF(cu_warp8u, i);
    REF(cu_warp32f, i);
    REF(roi, i);
    REF(img, i);
    stream_fork.waitEvent(event);
    cu_img.upload(img, stream_fork);
    cv::cuda::remap(cu_img, cu_warp8u, cu_mapx, cu_mapy, cv::INTER_LINEAR,
                     cv::BORDER_CONSTANT, 0, stream_fork);

    cu_warp8u.convertTo(cu_warp32f, CV_32FC3, stream_fork);
    cv::cuda::multiply(cu_warp32f, cu_mask, cu_warp32f, 1, CV_32FC3,
                        stream_fork);
    cu_warp32f.convertTo(cu_warp8u, CV_8UC3, stream_fork);
    event_fork.record(stream_fork);
  }
  for (int i = 0; i < frames; i++) {
    REF(event_fork, i);
    REF(cu_warp8u, i);
    REF(roi, i);
    stream.waitEvent(event_fork);
    cv::cuda::add(cu_canvas(roi), cu_warp8u, cu_canvas(roi), cv::noArray(),
                   CV_8UC3, stream);
  }
  cu_canvas.download(result, stream);
  stream.waitForCompletion();
  TOCK(operator);
}
#endif


#ifndef USE_CUDA
void FisheyeAVM::operator()(const std::vector<cv::Mat> &imgs,
                            cv::Mat &result) const {
  TICK(operator);
  canvas.setTo(cv::Scalar(0, 0, 0));
  for (int i = 0; i < frames; i++) {

    REF(mask, i);
    REF(mapx, i);
    REF(mapy, i);
    REF(warp8u, i);
    REF(warp32f, i); 
    REF(img_tem, i);
    REF(roi, i);
    REF(img, i);
    cv::remap(img, warp8u, mapx, mapy, cv::INTER_LINEAR); 
    warp8u.convertTo(warp32f, CV_32FC3 );

    cv::multiply(warp32f, mask, warp32f, 1, CV_32FC3);
    warp32f.convertTo(warp8u, CV_8UC3);
  }
  for (int i = 0; i < frames; i++) {
    REF(warp8u, i);
    REF(roi, i);
    cv::add(canvas(roi), warp8u, canvas(roi), cv::noArray(), CV_8UC3);
  }
  result = canvas;
  TOCK(operator);
}

cv::Mat FisheyeAVM::operator()(const std::vector<cv::Mat> &imgs) const {
  (*this)(imgs, canvas);
  return canvas;
}
#endif