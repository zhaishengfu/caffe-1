#include "caffe/layers/cross_entropy.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

  template <typename Dtype>
  void CrossEntropyLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
					 const vector<Blob<Dtype>*>& top) {
    top[0]->ReshapeLike(*bottom[0]);
  }

  template <typename Dtype>
  void CrossEntropyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
					     const vector<Blob<Dtype>*> &top) {
    const int count = bottom[0]->count();
    const int num = bottom[0]->num();
    const Dtype* input_data = bottom[0]->cpu_data();
    const Dtype* target = bottom[1]->cpu_data();
    const int dim = count / num;
    for (int i=0;i<num;i++) {
      Dtype loss = 0;
      for (int j=i*dim;j<(i+1)*dim;j++) {
	loss -= input_data[j] * (target[j] - (input_data[j] >= 0)) -
	  log(1 + exp(input_data[j] - 2 * input_data[j] * (input_data[j] >= 0)));
      }
      top[0]->mutable_cpu_data()[i] = loss / num;
    }
  }

INSTANTIATE_CLASS(CrossEntropyLayer);
REGISTER_LAYER_CLASS(CrossEntropy);
  
}
