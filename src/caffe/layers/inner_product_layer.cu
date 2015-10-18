#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include <iostream>
#include <string>     // std::string, std::stoi
#include <fstream>

namespace caffe {

template <typename Dtype>
void InnerProductLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

    // STEFAN HACK: I am going to query a file to see whether this is the last layer
  // PROPER VERSION: Make a new layer which is like IP layer but does communication
  LOG(INFO) << "***FORWARD!";
  // Read file and see if we are at the last layer
  std::string folder = std::getenv("FOLDER");
  std::string phase  = std::getenv("PHASE");
  std::string mode   = std::getenv("MODE"); // caffe, lr, mtlr
  std::string init   = std::getenv("INIT");
  std::ifstream check_last_layer_file;
  check_last_layer_file.open ((folder + "/last_layer_flag.txt").c_str());
  std::string last_layer_str;
  check_last_layer_file >> last_layer_str;
  bool last_layer = false;
  if (last_layer_str == "1") {
    last_layer = true;
  } else {
    assert(last_layer_str == "0");
  }
  check_last_layer_file.close();
  // LOG(INFO) << "****last layer=" << last_layer;
  last_layer = mode != "caffe" && last_layer;

  // We also need to know the number of images in the data set
  long DATASET_SIZE = 0;
  std::ifstream dataset_size_file;
  std::string dataset_size_filename;
  if (phase == "train" || phase == "val") {
    dataset_size_filename = folder + "/n_train";
  } else if (phase == "test") {
    dataset_size_filename = folder + "/n_test";
  } else {
    exit(1);
  }

  dataset_size_file.open (dataset_size_filename.c_str());
  std::string dataset_size_str;
  dataset_size_file >> dataset_size_str;
  DATASET_SIZE = atoi(dataset_size_str.c_str());
  dataset_size_file.close();
  assert(DATASET_SIZE > 0);

    // we need to know if this is the first run to decide whether to initialize weights
  bool first_run = false;
  if (last_layer_str == "1" && init == "1") {
    std::fstream first_run_file;
    first_run_file.open ((folder + "/first_run_file.txt").c_str());
    std::string first_run_str;
    first_run_file >> first_run_str;
    first_run = (first_run_str == "1");
    first_run_file.close();
    first_run_file.open((folder + "/first_run_file.txt").c_str());
    first_run_file << "0";
    first_run_file.close();
    // LOG(INFO) << "****first_run=" << first_run;
  }

  // If not last layer, continue as normal
  if (!last_layer) {

      const Dtype* bottom_data = bottom[0]->gpu_data();
      Dtype* top_data = top[0]->mutable_gpu_data();
      const Dtype* weight = this->blobs_[0]->gpu_data();
      if (M_ == 1) {
        caffe_gpu_gemv<Dtype>(CblasNoTrans, N_, K_, (Dtype)1.,
                             weight, bottom_data, (Dtype)0., top_data);
        if (bias_term_)
          caffe_gpu_axpy<Dtype>(N_, bias_multiplier_.cpu_data()[0],
                                this->blobs_[1]->gpu_data(), top_data);
      } else {
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
                              bottom_data, weight, (Dtype)0., top_data);
        if (bias_term_)
          caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
                                bias_multiplier_.gpu_data(),
                                this->blobs_[1]->gpu_data(), (Dtype)1., top_data);
      }

  }
  // Otherwise, now we need to write these features (bottom[0]->cpu_data()) to a file
  else {
    std::ofstream fw_output_file;
    fw_output_file.open ((folder + "/cnn_features.txt").c_str());

    const Dtype* bottom_data = bottom[0]->cpu_data();
    const int batch_size = bottom[0]->shape()[0];
    const int example_size = bottom[0]->shape()[1];

    // Also read a file to find the current batch ID
    std::ifstream current_img_file;
    current_img_file.open ((folder + "/current_img.txt").c_str());
    std::string current_img_str;
    current_img_file >> current_img_str;
    int current_img = atoi(current_img_str.c_str());
    current_img_file.close();

    // For img in batch
    for (long batch_idx=0; batch_idx < batch_size; ++batch_idx){
      // First, write out the filename
      fw_output_file << (batch_idx + current_img) % DATASET_SIZE << ",";
      // Now, write out the data
      for (long feature_idx=0; feature_idx < example_size - 1; ++feature_idx){
        fw_output_file << bottom_data[example_size*batch_idx + feature_idx] << ",";
      }
      fw_output_file << bottom_data[example_size*batch_idx + example_size - 1];
      fw_output_file << "\n";
    }
    fw_output_file.close();

    std::string weight_file = folder + "/weight_file";
    std::string label_file = folder + "/label_file";
    std::string feature_file = folder + "/cnn_features.txt";
    std::string gradient_file = folder + "/cnn_gradients.txt";

    int n_classes = this->blobs_[0]->shape()[0];

    // initialize weights for the first run
    if (first_run && phase == "train") {
      LOG(INFO) << "***NOTICE: INITIALIZING WEIGHTS";
      std::ofstream weight_file_out(weight_file.c_str());
      const Dtype* weight = this->blobs_[0]->cpu_data();
      for (int i = 0; i < n_classes; i++) {
        weight_file_out << weight[i * n_classes];
        for (int j = 1; j < this->blobs_[0]->shape()[1]; j++) {
          weight_file_out << "," << weight[i * example_size + j];
        }
        weight_file_out << "," << this->blobs_[1]->cpu_data()[i];
        weight_file_out << "\n";
      }
      weight_file_out.close();
    }

    // Now call the executable
    int ret = 0;
    if (phase == "train") {
      std::stringstream ss;
      ss << "python mtlr.py " << feature_file << " " << weight_file << " " << label_file << " "
        << DATASET_SIZE << " " << n_classes << " " << example_size << " 0.01 0.01 1 > " << gradient_file;
      LOG(INFO) << ss.str();
      ret = system(ss.str().c_str());
      if (ret != 0) exit(1);
    } else {
      std::stringstream ss;
      ss << "python predict_mtlr.py " << feature_file << " " << weight_file << " " << n_classes << " " << example_size << " > " << folder + "/cnn_probabilities.txt";
      LOG(INFO) << ss.str();
      ret = system(ss.str().c_str());
      if (ret != 0) exit(1);
      // set last layer flag to 0
      std::ofstream check_last_layer_file;
      check_last_layer_file.open ((folder + "/last_layer_flag.txt").c_str());
      check_last_layer_file << "0";
      check_last_layer_file.close();
    }

    // Write back the new ID
    std::ofstream current_img_file2;
    current_img_file2.open ((folder + "/current_img.txt").c_str());
    current_img_file2 << ( current_img + batch_size ) % DATASET_SIZE;
    current_img_file2.close();
  }

}

template <typename Dtype>
void InnerProductLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  // STEFAN HACK: Like in FW pass I am going to query a file to see whether this is the last layer
  // PROPER VERSION: Make a new layer which is like IP layer but does communication
  LOG(INFO) << "***BACKWARD!";
  std::string folder = std::getenv("FOLDER");
  std::string mode   = std::getenv("MODE"); // caffe, lr, mtlr
  // Read file and see if we are at the last layer
  std::fstream check_last_layer_file;
  check_last_layer_file.open ((folder + "/last_layer_flag.txt").c_str());
  std::string last_layer_str;
  check_last_layer_file >> last_layer_str;
  check_last_layer_file.close();
  bool last_layer = false;
  if (last_layer_str == "1") {
    last_layer = true;
  }
  // LOG(INFO) << "****last layer=" << last_layer;
  last_layer = (mode != "caffe") && last_layer;

  // If not last layer, continue as normal
  if (!last_layer) {

      if (this->param_propagate_down_[0]) {
        const Dtype* top_diff = top[0]->gpu_diff();
        const Dtype* bottom_data = bottom[0]->gpu_data();
        // Gradient with respect to weight
        caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
            top_diff, bottom_data, (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
      }
      if (bias_term_ && this->param_propagate_down_[1]) {
        const Dtype* top_diff = top[0]->gpu_diff();
        // Gradient with respect to bias
        caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
            bias_multiplier_.gpu_data(), (Dtype)1.,
            this->blobs_[1]->mutable_gpu_diff());
      }
      if (propagate_down[0]) {
        const Dtype* top_diff = top[0]->gpu_diff();
        // Gradient with respect to bottom data
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
            top_diff, this->blobs_[0]->gpu_data(), (Dtype)0.,
            bottom[0]->mutable_gpu_diff());
      }

  }
  // Otherwise, now we need to update gradients
  else {

    // First make buffer for input gradients
    const int batch_size = bottom[0]->shape()[0];
    const int example_size = bottom[0]->shape()[1];

    // Open file and read features
    std::ifstream dd_output_file;
    dd_output_file.open ((folder + "/cnn_gradients.txt").c_str());
    for (long grad_idx=0; grad_idx < batch_size*example_size; ++grad_idx){
      dd_output_file >> bottom[0]->mutable_cpu_diff()[grad_idx];
    }
    dd_output_file.close();

    cudaMemcpy(bottom[0]->mutable_gpu_diff(), bottom[0]->cpu_diff(), sizeof(Dtype) * example_size, cudaMemcpyHostToDevice);

  }

  // Finally, reset last layer flag
  check_last_layer_file.open ((folder + "/last_layer_flag.txt").c_str());
  check_last_layer_file << "0";
  check_last_layer_file.close();

}

INSTANTIATE_LAYER_GPU_FUNCS(InnerProductLayer);

}  // namespace caffe
