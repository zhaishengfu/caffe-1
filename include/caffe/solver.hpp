#ifndef CAFFE_OPTIMIZATION_SOLVER_HPP_
#define CAFFE_OPTIMIZATION_SOLVER_HPP_

#include <string>
#include <vector>

#include "caffe/net.hpp"

#include <cmaes.h>
using namespace libcmaes;

namespace caffe {

template <typename Dtype>
class Solver {
 public:
  explicit Solver(const SolverParameter& param);
  explicit Solver(const string& param_file);
  void Init(const SolverParameter& param);
  void InitTrainNet();
  void InitTestNets();
  // The main entry of the solver function. In default, iter will be zero. Pass
  // in a non-zero iter number to resume training for a pre-trained net.
  virtual void Solve(const char* resume_file = NULL);
  inline void Solve(const string resume_file) { Solve(resume_file.c_str()); }
  virtual ~Solver() {}
  inline shared_ptr<Net<Dtype> > net() { return net_; }
  inline const vector<shared_ptr<Net<Dtype> > >& test_nets() {
    return test_nets_;
  }

 protected:
  // PreSolve is run before any solving iteration starts, allowing one to
  // put up some scaffold.
  virtual void PreSolve() {}
  // Get the update value for the current iteration.
  virtual void ComputeUpdateValue() = 0;
  // The Solver::Snapshot function implements the basic snapshotting utility
  // that stores the learned net. You should implement the SnapshotSolverState()
  // function that produces a SolverState protocol buffer that needs to be
  // written to disk together with the learned net.
  void Snapshot();
  // The test routine
  void TestAll();
  void Test(const int test_net_id = 0);
  virtual void SnapshotSolverState(SolverState* state) = 0;
  // The Restore function implements how one should restore the solver to a
  // previously snapshotted state. You should implement the RestoreSolverState()
  // function that restores the state from a SolverState protocol buffer.
  void Restore(const char* resume_file);
  virtual void RestoreSolverState(const SolverState& state) = 0;

  SolverParameter param_;
  int iter_;
  shared_ptr<Net<Dtype> > net_;
  vector<shared_ptr<Net<Dtype> > > test_nets_;

  DISABLE_COPY_AND_ASSIGN(Solver);
};


template <typename Dtype>
class SGDSolver : public Solver<Dtype> {
 public:
  explicit SGDSolver(const SolverParameter& param)
      : Solver<Dtype>(param) {}
  explicit SGDSolver(const string& param_file)
      : Solver<Dtype>(param_file) {}

 protected:
  virtual void PreSolve();
  Dtype GetLearningRate();
  virtual void ComputeUpdateValue();
  virtual void SnapshotSolverState(SolverState * state);
  virtual void RestoreSolverState(const SolverState& state);
  // history maintains the historical momentum data.
  vector<shared_ptr<Blob<Dtype> > > history_;

  DISABLE_COPY_AND_ASSIGN(SGDSolver);
};

class CMAESSolver : public Solver<double> {
 public:
  explicit CMAESSolver(const SolverParameter& param)
    : Solver<double>(param) {
    int dim = 0;
    std::cout << "creating solver from params\n";
    std::vector<double> x0;
    for (int param_id=0;param_id<this->net_->params().size();++param_id)
      {
	int cdim = this->net_->params()[param_id]->count();
	const double *ddata = this->net_->params()[param_id]->cpu_data();
	for (int i=0;i<cdim;i++)
	  x0.push_back(ddata[i]);
	dim += cdim;
      }
    double sigma = 0.01;
    //TODO: optional gradient function.
    CMAParameters<> cmaparams(x0,sigma);
    cmaparams.set_str_algo("sepacmaes");
    cmaparams.set_sep();
    cmaparams.set_max_hist(2);
    cmaparams.set_quiet(false);
    //cmaparams.set_noisy();
    optim_ = ESOptimizer<CMAStrategy<ACovarianceUpdate>,CMAParameters<>>(objloss_,cmaparams);
    optim_.set_gradient_func(gradf_);
  }
  explicit CMAESSolver(const string& param_file)
    : Solver<double>(param_file) {
    int dim = 0;
    std::cout << "creating solver from file\n";
    std::vector<double> x0;
    for (int param_id=0;param_id<this->net_->params().size();++param_id)
      {
	int cdim = this->net_->params()[param_id]->count();
	const double *ddata = this->net_->params()[param_id]->cpu_data();
	for (int i=0;i<cdim;i++)
	  x0.push_back(ddata[i]);
	dim += cdim;
      }
    double sigma = 0.01;
    //TODO: optional gradient function.
    CMAParameters<> cmaparams(x0,sigma);
    cmaparams.set_str_algo("sepacmaes");
    cmaparams.set_sep();
    cmaparams.set_max_hist(2);
    optim_ = ESOptimizer<CMAStrategy<ACovarianceUpdate>,CMAParameters<>>(objloss_,cmaparams);
    //optim_.set_gradient_func(gradf_);
  }

 protected:
  virtual void PreSolve();
  //double GetLearningRate();
  virtual void ComputeUpdateValue();
  virtual void SnapshotSolverState(SolverState * state);
  virtual void RestoreSolverState(const SolverState& state);
  // history maintains the historical momentum data.
  vector<shared_ptr<Blob<double> > > history_;

  ESOptimizer<CMAStrategy<ACovarianceUpdate>,CMAParameters<>> optim_;

  FitFunc objloss_ = [this](const double *x, const int N)
  {
    double *pos = const_cast<double*>(x);
    vector<shared_ptr<Blob<double> > >& net_params = this->net_->params();
    for (int param_id=0;param_id<net_params.size();++param_id) {
      caffe_copy(net_params[param_id]->count(),
		 pos,
		 net_params[param_id]->mutable_cpu_data());
      pos += net_params[param_id]->count();
    }
    double loss;
    std::vector<Blob<double>*> bottom_vec; // dummy.
    net_->Forward(bottom_vec,&loss);
    return static_cast<double>(loss);
  };

  GradFunc gradf_ = [this](const double *x, const int N)
  {
    dVec grad(N);
    double *pos = const_cast<double*>(x);
    //TODO: forward / backward pass on data.
    vector<shared_ptr<Blob<double> > >& net_params = this->net_->params();
    for (int param_id=0;param_id<net_params.size();++param_id) {
      caffe_copy(net_params[param_id]->count(),
		 pos,
		 net_params[param_id]->mutable_cpu_data());
      pos += net_params[param_id]->count();
    }

    std::vector<Blob<double>*> bottom_vec; // dummy.
    double loss = net_->ForwardBackward(bottom_vec);

    // fillup gradient data
    pos = grad.data();
    net_params = this->net_->params();
    for (int param_id=0;param_id<net_params.size();++param_id) {
      caffe_copy(net_params[param_id]->count(),
		 net_params[param_id]->cpu_diff(),
		 pos);
      pos += net_params[param_id]->count();
    }
    
    return grad;
  };
  
  DISABLE_COPY_AND_ASSIGN(CMAESSolver);
};

}  // namespace caffe

#endif  // CAFFE_OPTIMIZATION_SOLVER_HPP_
