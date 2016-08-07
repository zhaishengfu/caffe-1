#include <vector>

#include "gtest/gtest.h"

#include "caffe/sparse_blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/sparse_slice_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

#include <iostream>

namespace caffe {

template <typename TypeParam>
class SparseSliceLayerTest : public CPUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  SparseSliceLayerTest()
    : blob_bottom_(new SparseBlob<Dtype>(1, 1, 8)),
        blob_top_0_(new SparseBlob<Dtype>()),
      blob_top_1_(new SparseBlob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values up
    std::vector<Dtype> init_data = {10,-2,3,9,3,7,8,7};
    Dtype* data = blob_bottom_->mutable_cpu_data();
    for (size_t i=0;i<init_data.size();i++)
      data[i] = init_data[i];

    std::vector<int> init_ind = {1,5,1,2,6,2,3,4};
    int* indices = blob_bottom_->mutable_cpu_indices();
    for (size_t i=0;i<init_ind.size();i++)
      indices[i] = init_ind[i];

    std::vector<int> init_ptr = {1,3,6,9};
    int* ptr = blob_bottom_->mutable_cpu_ptr();
    for (size_t i=0;i<init_ptr.size();i++)
      ptr[i] = init_ptr[i];
    
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_1_.push_back(blob_top_0_);
    blob_top_vec_1_.push_back(blob_top_1_);
  }

  virtual ~SparseSliceLayerTest() {
    delete blob_top_0_; delete blob_top_1_;
    delete blob_bottom_;
  }

  SparseBlob<Dtype>* const blob_bottom_;
  SparseBlob<Dtype>* const blob_top_0_;
  SparseBlob<Dtype>* const blob_top_1_;
  vector<Blob<Dtype>*> blob_top_vec_0_, blob_top_vec_1_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
};

template <typename TypeParam>
class SparseSliceLayerTest2 : public CPUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  SparseSliceLayerTest2()
    : blob_bottom_(new SparseBlob<Dtype>(1, 1, 11)),
        blob_top_0_(new SparseBlob<Dtype>()),
      blob_top_1_(new SparseBlob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    std::vector<Dtype> init_data = {3,8,7,5,8,9,9,13,4,2,-1};
    Dtype* data = blob_bottom_->mutable_cpu_data();
    for (size_t i=0;i<init_data.size();i++)
      data[i] = init_data[i];

    std::vector<int> init_ind = {1,3,4,5,2,4,5,6,2,5,6};
    int* indices = blob_bottom_->mutable_cpu_indices();
    for (size_t i=0;i<init_ind.size();i++)
      indices[i] = init_ind[i];

    std::vector<int> init_ptr = {1,5,9,12};
    int* ptr = blob_bottom_->mutable_cpu_ptr();
    for (size_t i=0;i<init_ptr.size();i++)
      ptr[i] = init_ptr[i];
    
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_1_.push_back(blob_top_0_);
    blob_top_vec_1_.push_back(blob_top_1_);
  }

  virtual ~SparseSliceLayerTest2() {
    delete blob_top_0_; delete blob_top_1_;
    delete blob_bottom_;
  }

  SparseBlob<Dtype>* const blob_bottom_;
  SparseBlob<Dtype>* const blob_top_0_;
  SparseBlob<Dtype>* const blob_top_1_;
  vector<Blob<Dtype>*> blob_top_vec_0_, blob_top_vec_1_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
};

TYPED_TEST_CASE(SparseSliceLayerTest, TestDtypesAndDevices);
TYPED_TEST_CASE(SparseSliceLayerTest2, TestDtypesAndDevices);

TYPED_TEST(SparseSliceLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_slice_param()->add_slice_point(5);
  SparseSliceLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_1_);
  EXPECT_EQ(6,this->blob_top_0_->nnz());
  EXPECT_EQ(2,this->blob_top_1_->nnz());
}

TYPED_TEST(SparseSliceLayerTest, TestForward1) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_slice_param()->add_slice_point(5);
  SparseSliceLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_1_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_1_);

  std::vector<Dtype> ref_data0 = {10,3,9,7,8,7};  
  const Dtype* data0 = this->blob_top_0_->cpu_data();
  for (size_t i=0;i<ref_data0.size();i++)
    EXPECT_EQ(ref_data0[i],data0[i]);

  std::vector<Dtype> ref_data1 = {-2,3};
  const Dtype* data1 = this->blob_top_1_->cpu_data();
  for (size_t i=0;i<ref_data1.size();i++)
    EXPECT_EQ(ref_data1[i],data1[i]);

  std::vector<int> ref_ind0 = {1,1,2,2,3,4};
  const int* ind0 = this->blob_top_0_->cpu_indices();
  for (size_t i=0;i<ref_ind0.size();i++)
    EXPECT_EQ(ref_ind0[i],ind0[i]);
  
  std::vector<int> ref_ind1 = {1,2};
  const int* ind1 = this->blob_top_1_->cpu_indices();
  for (size_t i=0;i<ref_ind1.size();i++)
    EXPECT_EQ(ref_ind1[i],ind1[i]);
  
  std::vector<int> ref_ptr0 = {1,2,4,7};
  const int* ptr0 = this->blob_top_0_->cpu_ptr();
  for (size_t i=0;i<ref_ptr0.size();i++)
    EXPECT_EQ(ref_ptr0[i],ptr0[i]);

  std::vector<int> ref_ptr1 = {1,2,3};
  const int* ptr1 = this->blob_top_1_->cpu_ptr();
  for (size_t i=0;i<ref_ptr1.size();i++)
    EXPECT_EQ(ref_ptr1[i],ptr1[i]);
}

TYPED_TEST(SparseSliceLayerTest2, TestForward2) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_slice_param()->add_slice_point(3);
  SparseSliceLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_1_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_1_);

  std::vector<Dtype> ref_data0 = {3,8,4};  
  const Dtype* data0 = this->blob_top_0_->cpu_data();
  for (size_t i=0;i<ref_data0.size();i++)
    EXPECT_EQ(ref_data0[i],data0[i]);

  std::vector<Dtype> ref_data1 = {8,7,5,9,9,13,2,-1};
  const Dtype* data1 = this->blob_top_1_->cpu_data();
  for (size_t i=0;i<ref_data1.size();i++)
    EXPECT_EQ(ref_data1[i],data1[i]);

  std::vector<int> ref_ind0 = {1,2,2};
  const int* ind0 = this->blob_top_0_->cpu_indices();
  for (size_t i=0;i<ref_ind0.size();i++)
    EXPECT_EQ(ref_ind0[i],ind0[i]);
  
  std::vector<int> ref_ind1 = {1,2,3,2,3,4,3,4};
  const int* ind1 = this->blob_top_1_->cpu_indices();
  for (size_t i=0;i<ref_ind1.size();i++)
    EXPECT_EQ(ref_ind1[i],ind1[i]);
  
  std::vector<int> ref_ptr0 = {1,2,3,4};
  const int* ptr0 = this->blob_top_0_->cpu_ptr();
  for (size_t i=0;i<ref_ptr0.size();i++)
    EXPECT_EQ(ref_ptr0[i],ptr0[i]);
  
  std::vector<int> ref_ptr1 = {1,4,7,9};
  const int* ptr1 = this->blob_top_1_->cpu_ptr();
  for (size_t i=0;i<ref_ptr1.size();i++)
    EXPECT_EQ(ref_ptr1[i],ptr1[i]);
}

}  // namespace caffe
