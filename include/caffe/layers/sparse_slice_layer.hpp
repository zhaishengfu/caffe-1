#ifndef CAFFE_SPARSE_SLICE_LAYER_HPP_
#define CAFFE_SPARSE_SLICE_LAYER_HPP_

#include <vector>

#include "caffe/sparse_blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {


  /**
   * @brief Takes a SparseBlob and slices it along either the num or channel dimension,
   *        outputting multiple sliced Blob results.
   */
  template <typename Dtype>
class SparseSliceLayer : public Layer<Dtype> {
 public:
  explicit SparseSliceLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Slice"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
    {
      CAFFE_NOT_IMPLEMENTED;
    }
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
    {
      CAFFE_NOT_IMPLEMENTED;
    }

  int count_;
  int num_slices_;
  int slice_size_;
  int slice_axis_;
  vector<int> slice_point_;
};
  
}

#endif
