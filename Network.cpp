#include "Network.h"

std::vector<Tensor3D> Network::forward(const Tensor3D& input) const
{
	std::vector<Tensor3D> tensorStack({ input });

	for (const Layer& layer : mLayers)
		tensorStack.push_back(convolution(relu(tensorStack.back()), layer.kernels, layer.stride, layer.padding));

	return tensorStack;
}

// Compute the gradient
void Network::backward(const std::vector<Tensor3D>& tensorStack, const Eigen::VectorXd& target, std::vector<std::vector<Tensor3D>>& weightsGradient, std::vector<std::vector<double>>& biasesGradient, double huberLossDelta)
{
	double loss(0);
	int l(tensorStack.size() - 1);

	weightsGradient.resize(l);
	biasesGradient.resize(l);

	std::vector<Tensor3D> deltas;

	for (const Tensor3D& tensor : tensorStack)
		deltas.push_back(Tensor3D(tensor.height(), tensor.width(), tensor.depth())); // We want the same structure as "predicted" but without copying

	for (int i(0); i < target.size(); ++i) {
		// Huber Loss

		deltas[l](0, 0, i) = tensorStack[l](0, 0, i) - target(i);

		if (std::abs(deltas[l](0, 0, i)) > huberLossDelta)
			deltas[l](0, 0, i) = huberLossDelta * (std::abs(deltas[l](0, 0, i)) / deltas[l](0, 0, i));
	}

	--l;

	while (l >= 0) {
		size_t kernelHeight = mLayers[l].kernels[0].weights.height(),
			   kernelWidth = mLayers[l].kernels[0].weights.width(),
			   nbKernels = mLayers[l].kernels.size(),
			   s = mLayers[l].stride,
			   p = mLayers[l].padding;

		weightsGradient[l].resize(nbKernels);
		biasesGradient[l].resize(nbKernels);

		// Compute deltas
		deltas[l] = deconvolution(deltas[l + 1], mLayers[l].kernels, mLayers[l].stride, mLayers[l].padding);

		for (size_t i(0); i < deltas[l].height(); ++i)
			for (size_t j(0); j < deltas[l].width(); ++j)
				for (size_t k(0); k < deltas[l].depth(); ++k)
					if (tensorStack[l](i, j, k) <= 0)
						deltas[l](i, j, k) = 0; // ReLU derivative is 0 if the input is <= 0

		// Computing weights gradient
		for (size_t k(0); k < deltas[l + 1].depth(); ++k) {
			weightsGradient[l][k] = Tensor3D(kernelHeight, kernelWidth, deltas[l].depth());

			for (size_t i(0); i < deltas[l + 1].height(); ++i) {
				for (size_t j(0); j < deltas[l + 1].width(); ++j) {
					for (size_t m(0); m < kernelHeight; ++m) {
						for (size_t n(0); n < kernelWidth; ++n) {
							int x = i * s + m - p,
								y = j * s + n - p;

							if (x < 0 || x >= deltas[l].height() || y < 0 || y >= deltas[l].width())
								continue;

							for (size_t c(0); c < deltas[l].depth(); ++c)
								weightsGradient[l][k](m, n, c) += deltas[l + 1](i, j, k) * std::max(tensorStack[l](x, y, c), 0.0);
						}
					}
				}
			}
		}

		// Computing biases gradient
		for (size_t k(0); k < deltas[l + 1].depth(); ++k) {
			biasesGradient[l][k] = 0;

			for (size_t i(0); i < deltas[l + 1].height(); ++i)
				for (size_t j(0); j < deltas[l + 1].width(); ++j)
					biasesGradient[l][k] += deltas[l + 1](i, j, k);
		}

		--l;
	}
}

void Network::applyGradient(const std::vector<std::vector<Tensor3D>>& weightsGrad, const std::vector<std::vector<double>>& biasesGrad, double momentumTerm, double smoothingTerm)
{
	for (size_t l(0); l < mLayers.size(); ++l) {
		size_t kernelHeight = mLayers[l].kernels[0].weights.height(),
			   kernelWidth = mLayers[l].kernels[0].weights.width(),
			   kernelDepth = mLayers[l].kernels[0].weights.depth(),
			   nbKernels = mLayers[l].kernels.size();

		for (size_t k(0); k < nbKernels; ++k) {
			_update(mLayers[l].kernels[k].bias, mLayers[l].kernels[k].biasAvgGrad, mLayers[l].kernels[k].biasAvgUpdate, biasesGrad[l][k], momentumTerm, smoothingTerm);

			for (size_t m(0); m < kernelHeight; ++m)
				for (size_t n(0); n < kernelWidth; ++n)
					for (size_t c(0); c < kernelDepth; ++c)
						_update(mLayers[l].kernels[k].weights(m, n, c), mLayers[l].kernels[k].weightsAvgGrad(m, n, c), mLayers[l].kernels[k].weightsAvgUpdate(m, n, c), weightsGrad[l][k](m, n, c), momentumTerm, smoothingTerm);
		}
	}
}

void Network::addLayer(size_t nbKernels, size_t kernelHeight, size_t kernelWidth, size_t kernelChannels, size_t stride, size_t padding)
{
	mLayers.push_back({ std::vector<Kernel>(nbKernels, { Tensor3D(kernelHeight, kernelWidth, kernelChannels), 0.0 }), stride, padding });
	

	std::mt19937 generator(std::random_device{}());
	std::normal_distribution<double> rand(0.0, 1.0);

	for (int k(0); k < nbKernels; ++k)
		for (size_t m(0); m < kernelHeight; ++m)
			for (size_t n(0); n < kernelWidth; ++n)
				for (size_t c(0); c < kernelChannels; ++c)
					mLayers.back().kernels[k].weights(m, n, c) = rand(generator) * (2.0 / double(kernelHeight * kernelWidth * kernelChannels));
}

Layer& Network::layer(size_t l)
{
	return mLayers[l];
}

const Layer& Network::layer(size_t l) const
{
	return const_cast<Network*>(this)->layer(l);
}

void Network::_update(double& x, double& avgGrad, double& avgUpdate, double grad, double momentumTerm, double smoothingTerm)
{
	avgGrad = momentumTerm * avgGrad + (1 - momentumTerm) * grad * grad;
	double update = -std::sqrt((avgUpdate + smoothingTerm) / (avgGrad + smoothingTerm)) * grad;
	avgUpdate = momentumTerm * avgUpdate + (1 - momentumTerm) * update * update;

	x += update;
}
