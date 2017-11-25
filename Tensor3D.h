#ifndef TENSOR3D_H
#define TENSOR3D_H

#include <vector>
#include <random>
#include <algorithm>
#include <Eigen\Dense>

struct Kernel;

class Tensor3D
{
public:
	Tensor3D();
	Tensor3D(const std::vector<Eigen::MatrixXd>&);
	Tensor3D(size_t, size_t, size_t);

	size_t width() const;
	size_t height() const;
	size_t depth() const;

	Eigen::MatrixXd& operator[](size_t);
	const Eigen::MatrixXd& operator[](size_t) const;

	double& operator()(size_t, size_t, size_t);
	const double& operator()(size_t, size_t, size_t) const;

private:
	std::vector<Eigen::MatrixXd> mChannels;
};

struct Kernel
{
	Kernel(const Tensor3D& w, double b) : 
		weights(w), 
		bias(b), 
		weightsAvgGrad(weights.height(), weights.width(), weights.depth()),
		weightsAvgUpdate(weights.height(), weights.width(), weights.depth()),
		biasAvgGrad(0.0),
		biasAvgUpdate(0.0)
	{ }

	Tensor3D weights;
	double bias;

	Tensor3D weightsAvgGrad, weightsAvgUpdate;
	double biasAvgGrad, biasAvgUpdate;
};

Tensor3D convolution(const Tensor3D&, const std::vector<Kernel>&, int, int);
Tensor3D deconvolution(const Tensor3D&, const std::vector<Kernel>&, int, int);
Tensor3D relu(const Tensor3D&);

#endif // TENSOR3D_H
