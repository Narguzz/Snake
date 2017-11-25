#include "Tensor3D.h"

Tensor3D::Tensor3D() : Tensor3D(0, 0, 0)
{
}

Tensor3D::Tensor3D(const std::vector<Eigen::MatrixXd>& channels) :
		mChannels(channels)
{
}

// Weights are given a default value of 0
Tensor3D::Tensor3D(size_t height, size_t width, size_t depth) :
	mChannels(depth, Eigen::MatrixXd::Zero(height, width))
{
}

size_t Tensor3D::width() const
{
	if (!depth())
		return 0;

	return mChannels[0].cols();
}

size_t Tensor3D::height() const
{
	if (!depth())
		return 0;

	return mChannels[0].rows();
}

size_t Tensor3D::depth() const
{
	return mChannels.size();
}

Eigen::MatrixXd& Tensor3D::operator[](size_t k)
{
	return mChannels[k];
}

const Eigen::MatrixXd& Tensor3D::operator[](size_t k) const
{
	return const_cast<Tensor3D*>(this)->operator[](k);
}

double& Tensor3D::operator()(size_t i, size_t j, size_t k)
{
	return mChannels[k](i, j);
}

const double& Tensor3D::operator()(size_t i, size_t j, size_t k) const
{
	return const_cast<Tensor3D*>(this)->operator()(i, j, k);
}

// Assuming all kernels have the same size
Tensor3D convolution(const Tensor3D& input, const std::vector<Kernel>& kernels, int stride, int padding)
{
	int kernelHeight(kernels[0].weights.height()),
		kernelWidth(kernels[0].weights.width()),
		outputHeight((input.height() - kernelHeight + 2 * padding) / stride + 1),
		outputWidth((input.width() - kernelWidth + 2 * padding) / stride + 1);

	Eigen::MatrixXd inputCols(kernelHeight * kernelWidth * input.depth(), outputHeight * outputWidth);
	Eigen::MatrixXd weightsRows(kernels.size(), kernelHeight * kernelWidth * input.depth());

	// Initialize the input matrix
	for (size_t m(0); m < kernelHeight; ++m) {
		for (size_t n(0); n < kernelWidth; ++n) {
			for (size_t c(0); c < input.depth(); ++c) {
				size_t weightIndex = c * kernelHeight * kernelWidth + n * kernelHeight + m;

				for (size_t k(0); k < kernels.size(); ++k)
					weightsRows(k, weightIndex) = kernels[k].weights(m, n, c);

				for (size_t i(0); i < outputHeight; ++i)
					for (size_t j(0); j < outputWidth; ++j) {
						int x = stride * i + m - padding, 
							y = stride * j + n - padding;

						if (x < 0 || x >= input.height() || y < 0 || y >= input.width())
							inputCols(weightIndex, j * outputHeight + i) = 0; // Zero-padding
						else
							inputCols(weightIndex, j * outputHeight + i) = input(x, y, c);
					}
			}
		}
	}

	// Compute the matrix product
	Eigen::MatrixXd outputMatrix(weightsRows * inputCols); // dimensions : kernels.size() x (outputHeight * outputWidth)

	Tensor3D output(outputHeight, outputWidth, kernels.size());

	for (size_t i(0); i < outputHeight; ++i)
		for (size_t j(0); j < outputWidth; ++j)
			for (size_t k(0); k < kernels.size(); ++k)
				output(i, j, k) = outputMatrix(k, j * outputHeight + i) + kernels[k].bias;

	return output;
}

Tensor3D deconvolution(const Tensor3D& output, const std::vector<Kernel>& kernels, int stride, int padding)
{
	int kernelHeight(kernels[0].weights.height()),
		kernelWidth(kernels[0].weights.width()),
		kernelDepth(kernels[0].weights.depth()),
		inputHeight(stride * (output.height() - 1) + kernelHeight - 2 * padding),
		intputWidth(stride * (output.width() - 1) + kernelWidth - 2 * padding);

	Tensor3D input(inputHeight, inputHeight, kernelDepth);

	for (size_t k(0); k < kernels.size(); ++k) {
		for (size_t i(0); i < output.height(); ++i) {
			for (size_t j(0); j < output.width(); ++j) {
				for (size_t m(0); m < kernelHeight; ++m) {
					for (size_t n(0); n < kernelWidth; ++n) {
						int x = i * stride + m - padding, 
							y = j * stride + n - padding;

						if (x < 0 || x >= input.height() || y < 0 || y >= input.width())
							continue;

						for (size_t c(0); c < kernelDepth; ++c)
							input(x, y, c) += kernels[k].weights(m, n, c) * output(i, j, k);
					}
				}
			}
		}
	}

	return input;
}

Tensor3D relu(const Tensor3D& input)
{
	Tensor3D output(input);

	for (size_t i(0); i < output.depth(); ++i)
		output[i] = output[i].cwiseMax(0);

	return output;
}
