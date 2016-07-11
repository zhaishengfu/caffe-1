#include <algorithm>
#include <vector>

#include "caffe/layers/sparse_slice_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

  template <typename Dtype>
void SparseSliceLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
					 const vector<Blob<Dtype>*>& top) {
  const SliceParameter& slice_param = this->layer_param_.slice_param();
  CHECK(!(slice_param.has_axis() && slice_param.has_slice_dim()))
      << "Either axis or slice_dim should be specified; not both.";
  slice_point_.clear();
  std::copy(slice_param.slice_point().begin(),
      slice_param.slice_point().end(),
      std::back_inserter(slice_point_));
}

  //TODO: need nnz per slice
  // easier with a single slice point here as well
template <typename Dtype>
void SparseSliceLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
				      const vector<Blob<Dtype>*>& top) {
  const int num_axes = bottom[0]->num_axes();
  const SliceParameter& slice_param = this->layer_param_.slice_param();
  /*if (slice_param.has_slice_dim()) {
    slice_axis_ = static_cast<int>(slice_param.slice_dim());
    // Don't allow negative indexing for slice_dim, a uint32 -- almost
    // certainly unintended.
    CHECK_GE(slice_axis_, 0) << "casting slice_dim from uint32 to int32 "
        << "produced negative result; slice_dim must satisfy "
        << "0 <= slice_dim < " << kMaxBlobAxes;
    CHECK_LT(slice_axis_, num_axes) << "slice_dim out of range.";
    } else {*/
  slice_axis_ = bottom[0]->CanonicalAxisIndex(slice_param.axis());
  //}
  vector<int> top_shape = bottom[0]->shape();
  const int bottom_slice_axis = bottom[0]->shape(slice_axis_);
  num_slices_ = bottom[0]->count(0, slice_axis_);
  slice_size_ = bottom[0]->count(slice_axis_ + 1);
  
  vector<int> nnzs; //TODO
  SparseBlob<Dtype>* sparseBlob = dynamic_cast<SparseBlob<Dtype>*>(bottom[0]);
  if (!sparseBlob)
    LOG(FATAL) << "The bottom blob in the sparse slice layer is not sparse\n";    
  int offset_slice_axis = 0;
  const Dtype* bottom_data = sparseBlob->cpu_data();
  const int* indices = sparseBlob->cpu_indices();
  const int* ptr = sparseBlob->cpu_ptr();

  int slice_point = slice_point_[0];
  int nnz2 = 0;
  for (int i=0;i<sparseBlob->nnz();i++) {
    if (indices[i] < slice_point) {
      ++nnz2;
    }
  }
  int nnz1 = sparseBlob->nnz() - nnz2;

  // only a single slice point
  SparseBlob<Dtype>* sparseTopBlob1 = dynamic_cast<SparseBlob<Dtype>*>(top[0]);
  SparseBlob<Dtype>* sparseTopBlob2 = dynamic_cast<SparseBlob<Dtype>*>(top[1]);
  if (!sparseTopBlob1 || !sparseTopBlob2)
    LOG(FATAL) << "The top blob in the sparse slice layer is not sparse\n";
  Dtype* top_data1 = sparseTopBlob1->mutable_cpu_data();
  int* top_indices1 = sparseTopBlob1->mutable_cpu_indices();
  int* top_ptr1 = sparseTopBlob1->mutable_cpu_ptr();
  Dtype* top_data2 = sparseTopBlob2->mutable_cpu_data();
  int* top_indices2 = sparseTopBlob2->mutable_cpu_indices();
  int* top_ptr2 = sparseTopBlob2->mutable_cpu_ptr();
  
  int count = 0;
  if (slice_point_.size() != 0) {
    CHECK_EQ(slice_point_.size(), top.size() - 1);
    CHECK_LE(top.size(), bottom_slice_axis);
    int prev = 0;
    vector<int> slices;
    for (int i = 0; i < slice_point_.size(); ++i) {
      CHECK_GT(slice_point_[i], prev);
      slices.push_back(slice_point_[i] - prev);
      prev = slice_point_[i];
    }
    slices.push_back(bottom_slice_axis - prev);
    /*for (int i = 0; i < top.size(); ++i) {
      top_shape[slice_axis_] = slices[i];
      top[i]->Reshape(top_shape);
      count += top[i]->count();
      }*/
    top_shape[slice_axis_] = slices[0];
    sparseTopBlob1->Reshape(top_shape,nnz1);
    count += sparseTopBlob1->count();
    top_shape[slice_axis_] = slices[1];
    sparseTopBlob2->Reshape(top_shape,nnz2);
    count += sparseTopBlob2->count();
  } else {
    //TODO: error ?
    CHECK_EQ(bottom_slice_axis % top.size(), 0)
        << "Number of top blobs (" << top.size() << ") should evenly "
        << "divide input slice axis (" << bottom_slice_axis << ")";
    top_shape[slice_axis_] = bottom_slice_axis / top.size();
    for (int i = 0; i < top.size(); ++i) {
      top[i]->Reshape(top_shape);
      count += top[i]->count();
    }
  }
  CHECK_EQ(count, bottom[0]->count());
  if (top.size() == 1) {
    top[0]->ShareData(*bottom[0]);
    top[0]->ShareDiff(*bottom[0]);
  }
}

  // assumes a single slice point
template <typename Dtype>
void SparseSliceLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
					  const vector<Blob<Dtype>*>& top) {
  if (top.size() == 1) { return; }
  SparseBlob<Dtype>* sparseBlob = dynamic_cast<SparseBlob<Dtype>*>(bottom[0]);
  if (!sparseBlob)
    LOG(FATAL) << "The bottom blob in the sparse slice layer is not sparse\n";
      
  int offset_slice_axis = 0;
  const Dtype* bottom_data = sparseBlob->cpu_data();
  const int* indices = sparseBlob->cpu_indices();
  const int* ptr = sparseBlob->cpu_ptr();
  const int bottom_slice_axis = sparseBlob->shape(slice_axis_);
  /*for (int i = 0; i < top.size(); ++i) {
    Dtype* top_data = top[i]->mutable_cpu_data();
    const int top_slice_axis = top[i]->shape(slice_axis_);
    for (int n = 0; n < num_slices_; ++n) {
      const int top_offset = n * top_slice_axis * slice_size_;
      const int bottom_offset =
          (n * bottom_slice_axis + offset_slice_axis) * slice_size_;
      caffe_copy(top_slice_axis * slice_size_,
          bottom_data + bottom_offset, top_data + top_offset);
    }
    offset_slice_axis += top_slice_axis;
    }*/
  // only a single slice point
  SparseBlob<Dtype>* sparseTopBlob1 = dynamic_cast<SparseBlob<Dtype>*>(top[0]);
  SparseBlob<Dtype>* sparseTopBlob2 = dynamic_cast<SparseBlob<Dtype>*>(top[1]);
  if (!sparseTopBlob1 || !sparseTopBlob2)
    LOG(FATAL) << "The top blob in the sparse slice layer is not sparse\n";
  Dtype* top_data1 = sparseTopBlob1->mutable_cpu_data();
  int* top_indices1 = sparseTopBlob1->mutable_cpu_indices();
  int* top_ptr1 = sparseTopBlob1->mutable_cpu_ptr();
  Dtype* top_data2 = sparseTopBlob2->mutable_cpu_data();
  int* top_indices2 = sparseTopBlob2->mutable_cpu_indices();
  int* top_ptr2 = sparseTopBlob2->mutable_cpu_ptr();


  int slice_point = slice_point_[0];
  int ind_offset = sparseBlob->nnz() - slice_point;
  int top1_count = 0;
  int top2_count = 0;
  int ptr_idx = 0;
  int ptr2_idx = 0;
  int ptr_offset = 0;
  bool ptr_head = false;
  for (int i=0;i<sparseBlob->nnz();i++) {
    if (indices[i] >= slice_point) { // right side
      top_data2[top2_count] = bottom_data[i];
      top_indices2[top2_count] = indices[i] - ind_offset;
      if (!ptr_head)
	{
	  ptr_head = true;
	  top_ptr2[ptr2_idx] = i; // reconstruct the ptr table on the right
	  ptr2_idx++;
	}
      if (ptr[ptr_idx] < i && ptr[ptr_idx+1] > i) {
	top_ptr1[ptr_idx+1] -= 1; // decrement the next ptr on the left
	ptr_offset += 1;
      }
      ++top2_count;
    }
    else { // left side
      ptr_head = false;
      top_data1[top1_count] = bottom_data[i];
      top_indices1[top1_count] = indices[i];
      if (ptr[ptr_idx] == i) {
	if (i == 0)
	  top_ptr1[ptr_idx] = ptr[ptr_idx]; // 1
	++ptr_idx;
	top_ptr1[ptr_idx] = ptr[ptr_idx] - ptr_offset; // take original ptr location and apply offset
      }
      ++top1_count;
    }
  }
}

  //TODO: no need to backward, as slicing should occur right after input source
  /*template <typename Dtype>
void SparseSliceLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
					   const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0] || top.size() == 1) { return; }
  int offset_slice_axis = 0;
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const int bottom_slice_axis = bottom[0]->shape(slice_axis_);
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const int top_slice_axis = top[i]->shape(slice_axis_);
    for (int n = 0; n < num_slices_; ++n) {
      const int top_offset = n * top_slice_axis * slice_size_;
      const int bottom_offset =
          (n * bottom_slice_axis + offset_slice_axis) * slice_size_;
      caffe_copy(top_slice_axis * slice_size_,
          top_diff + top_offset, bottom_diff + bottom_offset);
    }
    offset_slice_axis += top_slice_axis;
  }
  }*/

#ifdef CPU_ONLY
  STUB_GPU_FORWARD(SparseSliceLayer, Forward);
#endif

INSTANTIATE_CLASS(SparseSliceLayer);
REGISTER_LAYER_CLASS(SparseSlice);

}
