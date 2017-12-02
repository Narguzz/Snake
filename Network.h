#ifndef NETWORK_H
#define NETWORK_H

#include <cmath>
#include <iostream>
#include "Tensor3D.h"

struct Layer
{
	std::vector<Kernel> kernels;
	size_t stride;
	size_t padding;
};

class Network
{
public:
	std::vector<Tensor3D> forward(const Tensor3D&) const;
	void backward(const std::vector<Tensor3D>&, const Eigen::VectorXd&, std::vector<std::vector<Tensor3D>>&, std::vector<std::vector<double>>&);

	void applyGradient(const std::vector<std::vector<Tensor3D>>&, const std::vector<std::vector<double>>&, double, double, double);

	void addLayer(size_t, size_t, size_t, size_t, size_t, size_t);

	Layer& layer(size_t);
	const Layer& layer(size_t) const;

private:
	void _update(double&, double&, double, double, double, double);

	std::vector<Layer> mLayers;
};

#endif // NETWORK_H
