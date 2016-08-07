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

// only supports a single slice point
template <typename Dtype>
void SparseSliceLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
				      const vector<Blob<Dtype>*>& top) {
  const SliceParameter& slice_param = this->layer_param_.slice_param();
  slice_axis_ = bottom[0]->CanonicalAxisIndex(slice_param.axis());
  vector<int> top_shape = bottom[0]->shape();
  const int bottom_slice_axis = bottom[0]->shape(slice_axis_); //TODO: beware
  num_slices_ = bottom[0]->count(0, slice_axis_);
  slice_size_ = bottom[0]->count(slice_axis_ + 1);
  
  SparseBlob<Dtype>* sparseBlob = dynamic_cast<SparseBlob<Dtype>*>(bottom[0]);
  if (!sparseBlob)
    LOG(FATAL) << "The bottom blob in the sparse slice layer is not sparse\n";    
  const int* indices = sparseBlob->cpu_indices();

  int slice_point = slice_point_[0];
  int nnz1 = 0;
  for (int i=0;i<sparseBlob->nnz();i++) {
    if (indices[i] < slice_point) {
      ++nnz1;
    }
  }
  int nnz2 = sparseBlob->nnz() - nnz1;
  
  // only a single slice point
  SparseBlob<Dtype>* sparseTopBlob1 = dynamic_cast<SparseBlob<Dtype>*>(top[0]);
  SparseBlob<Dtype>* sparseTopBlob2 = dynamic_cast<SparseBlob<Dtype>*>(top[1]);
  if (!sparseTopBlob1 || !sparseTopBlob2)
    LOG(FATAL) << "The top blob in the sparse slice layer is not sparse\n";
    
  int count = 0;
  if (slice_point_.size() != 0) {
    CHECK_EQ(slice_point_.size(), top.size() - 1);
    int prev = 0;
    vector<int> slices;
    for (int i = 0; i < slice_point_.size(); ++i) {
      CHECK_GT(slice_point_[i], prev);
      slices.push_back(slice_point_[i] - prev);
      prev = slice_point_[i];
    }
    slices.push_back(*std::max_element(indices,indices+sparseBlob->nnz()) - prev);
    top_shape[slice_axis_] = slices[0];
    sparseTopBlob1->Reshape(top_shape,nnz1);
    top_shape[slice_axis_] = slices[1];
    sparseTopBlob2->Reshape(top_shape,nnz2);
  } else {
    LOG(FATAL) << "SparseSliceLayer requires slice_point to be defined\n";
  }
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
  int ind_offset = slice_point - 1;
  int top1_count = 0;
  int top2_count = 0;
  int ptr_idx = 0;
  int ptr2_idx = 0;
  int ptr_offset = 0;
  bool ptr_head = false;
  for (int i=0;i<sparseBlob->nnz();i++) {
    int ind = indices[i];
    if (ind >= slice_point) { // right side
      top_data2[top2_count] = bottom_data[i];
      top_indices2[top2_count] = ind - ind_offset;
      //std::cerr << "right side i=" << i << " ptr_idx=" << ptr_idx << " / ptr[ptr_idx]=" << ptr[ptr_idx] << " / ptr[ptr_idx+1]=" << ptr[ptr_idx+1] << std::endl;
      if (!ptr_head)
	{
	  ptr_head = true;
	  top_ptr2[ptr2_idx] = top2_count + 1; // reconstruct the ptr table on the right -> TODO: offset ?
	  //std::cerr << "top_ptr2=" << top_ptr2[ptr2_idx] << " / ind=" << ind << std::endl;
	  ptr2_idx++;
	}
      if (ptr[ptr_idx] < i+1 && ptr[ptr_idx+1] > i+1) {
	top_ptr1[ptr_idx+1] -= 1; // decrement the next ptr on the left
	//std::cerr << "top_ptr1=" << top_ptr1[ptr_idx+1] << std::endl;
      }
      ptr_offset += 1;
      ++top2_count;
    }
    else { // left side
      ptr_head = false;
      top_data1[top1_count] = bottom_data[i];
      top_indices1[top1_count] = ind;
      //std::cerr << "i=" << i << " / ind=" << ind << " / ptr_idx=" << ptr_idx << " / ptr=" << ptr[ptr_idx] << std::endl;
      if (ptr[ptr_idx] == i+1) {
	if (i == 0)
	  top_ptr1[ptr_idx] = ptr[ptr_idx]; // 1
	else {
	  //std::cerr << "ptr[ptr_idx]=" << ptr[ptr_idx] << " / ptr_offset=" << ptr_offset << std::endl;
	  top_ptr1[ptr_idx] = ptr[ptr_idx] - ptr_offset; // take original ptr location and apply offset
	}
	++ptr_idx;
      }
      ++top1_count;
    }
  }
  //std::cerr << "ptr_idx=" << ptr_idx << std::endl;
  top_ptr1[ptr_idx] = sparseTopBlob1->nnz()+1;
  top_ptr2[ptr2_idx] = sparseTopBlob2->nnz()+1;
}

  // no need to backward, as slicing should occur right after input source
  /*template <typename Dtype>
void SparseSliceLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
					   const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
					  }
  }*/

#ifdef CPU_ONLY
  STUB_GPU_FORWARD(SparseSliceLayer, Forward);
#endif

INSTANTIATE_CLASS(SparseSliceLayer);
REGISTER_LAYER_CLASS(SparseSlice);

}
