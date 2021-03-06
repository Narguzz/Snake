#include "Agent.h"

Agent::Agent()
{
	/*Q.addLayer(32, 4, 4, 1, 2, 0);
	Q.addLayer(32, 2, 2, 32, 1, 0);
	Q.addLayer(128, 3, 3, 32, 1, 0);
	Q.addLayer(4, 1, 1, 128, 1, 0);*/
	
	Q.addLayer(16, 3, 3, 1, 1, 0);
	Q.addLayer(32, 3, 3, 16, 1, 0);
	Q.addLayer(64, 4, 4, 32, 1, 0);
	Q.addLayer(128, 1, 1, 64, 1, 0);
	Q.addLayer(256, 1, 1, 128, 1, 0);
	Q.addLayer(4, 1, 1, 256, 1, 0);
}

Direction Agent::optimalAction(const Tensor3D& state, std::vector<Tensor3D>& tensorStack) const
{
	tensorStack = Q.forward(state);

	size_t iMax(0);

	for (size_t i(0); i < tensorStack.back().depth(); ++i)
		if (tensorStack.back()(0, 0, i) > tensorStack.back()(0, 0, iMax))
			iMax = i;

	return Direction(iMax);
}

void Agent::train(size_t nbEpisodes, size_t batchSize, size_t replayMemorySize, double discountFactor, double epsStart, double epsEnd, double epsDecay, double learningRate, double momentumTerm, double smoothingTerm)
{
	Game game(10);

	size_t steps(0), episodes(0);
	int episodeSteps(0);

	std::mt19937 generator(std::random_device{}());
	std::uniform_real_distribution<double> rand(0.0, 1.0);
	std::uniform_int_distribution<size_t> randAction(0, 3);
	std::uniform_int_distribution<size_t> randAgent(0, 1);

	// Double DQN
	size_t agent = 0;

	Agent otherAgent;
	std::array<Agent*, 2> agents = { this, &otherAgent };
	std::array<std::string, 2> weightsPath = { "weights.txt", "weights2.txt" };

	std::array<ReplayMemory, 2> replayMemory{ ReplayMemory(replayMemorySize), ReplayMemory(replayMemorySize) };

	// Fill the replay memory
	for (size_t a(0); a < 2; ++a) {
		for (size_t i(0); i < replayMemorySize; ++i, ++episodeSteps) {
			Transition* t = new Transition(game, Direction(randAction(generator)));
			replayMemory[a].push(t, _priority(t->reward, 0.6, 1e-6)); // Mostly snake doing nothing

			if (t->isTerminal || episodeSteps >= 10 + 30 * game.score()) {
				game.initialize();
				episodeSteps = -1;
			}
		}
	}

	// Start the training
	while (episodes < nbEpisodes) {
		Transition* t = new Transition; // Current transition
		std::vector<Tensor3D> x;
		double epsilon(epsEnd + (epsStart - epsEnd) * exp(-1.0 * steps * epsDecay));

		t->state = Tensor3D(game.state());

		// Select an action to perform (epsilon-greedy policy)
		if (rand(generator) > epsilon)
			t->action = optimalAction(t->state, x);
		else
			t->action = Direction(randAction(generator));

		*t = Transition(game, t->action);

		if (t->isTerminal || episodeSteps >= 10 + 30 * game.score()) {
			++episodes;
			episodeSteps = -1;
			std::cout << episodes << " / " << nbEpisodes << ": " << game.score() << "\n";

			agents[0]->saveToFile(weightsPath[0]);
			agents[1]->saveToFile(weightsPath[1]);
			game.initialize();
		}

		replayMemory[agent].push(t, 100.0); // Big priority to ensure it will be sampled immediately
		std::vector<size_t> batch(replayMemory[agent].sample(batchSize));


		std::vector<std::vector<std::vector<Tensor3D>>> weightsGradients(batch.size());
		std::vector<std::vector<std::vector<double>>> biasesGradients(batch.size());

		// Compute gradients for the batch
		for (size_t i(0); i < batch.size(); ++i) {
			t = replayMemory[agent][batch[i]].transition;
			agents[agent]->optimalAction(t->state, x);

			// Compute target vector
			Eigen::VectorXd target(x.back().depth());

			for (size_t i(0); i < x.back().depth(); ++i)
				target(i) = x.back()(0, 0, i);

			target(t->action) = t->reward;

			if (!t->isTerminal) {
				Direction nextAction;
				std::vector<Tensor3D> nextX;

				nextAction = agents[agent]->optimalAction(t->nextState);
				agents[1 - agent]->optimalAction(t->nextState, nextX);

				target(t->action) += discountFactor * nextX.back()(0, 0, nextAction);
			}

			replayMemory[agent].setVal(batch[i], _priority(x.back()(0, 0, t->action) - target(t->action), 0.6, 1e-6)); // Update transition priority
			agents[agent]->Q.backward(x, target, weightsGradients[i], biasesGradients[i]);
		}

		// Average the gradients
		std::vector<std::vector<Tensor3D>> weightsGradient(weightsGradients[0]);
		std::vector<std::vector<double>> biasesGradient(biasesGradients[0]);

		for (size_t l(0); l < weightsGradient.size(); ++l) {
			size_t kernelHeight = weightsGradient[l][0].height(),
				   kernelWidth = weightsGradient[l][0].width(),
				   kernelDepth = weightsGradient[l][0].depth(),
				   nbKernels = weightsGradient[l].size();

			for (size_t k(0); k < nbKernels; ++k) {
				weightsGradient[l][k] = Tensor3D(kernelHeight, kernelWidth, kernelDepth);
				biasesGradient[l][k] = 0;

				for (size_t p(0); p < batch.size(); ++p) {
					biasesGradient[l][k] += biasesGradients[p][l][k] / batch.size();

					for (size_t m(0); m < kernelHeight; ++m)
						for (size_t n(0); n < kernelWidth; ++n)
							for (size_t c(0); c < kernelDepth; ++c)
								weightsGradient[l][k](m, n, c) += weightsGradients[p][l][k](m, n, c) / batch.size();
				}
			}
		}

		agents[agent]->Q.applyGradient(weightsGradient, biasesGradient, learningRate, momentumTerm, smoothingTerm);
		++steps;
		++episodeSteps;

		agent = randAgent(generator);
	}
}

void Agent::saveToFile(const std::string& path) const
{
	std::ofstream file;
	file.open(path, std::fstream::trunc);

	int kernelHeight(0), kernelWidth(0), kernelDepth(0);

	for (int l(0); l < Q.layers(); ++l) {
		kernelHeight = Q.layer(l).kernels[0].weights.height();
		kernelWidth = Q.layer(l).kernels[0].weights.width();
		kernelDepth = Q.layer(l).kernels[0].weights.depth();

		file << Q.layer(l).stride << "\n";
		file << Q.layer(l).padding << "\n";

		for (int k(0); k < Q.layer(l).kernels.size(); ++k) {
			for (int m(0); m < kernelHeight; ++m) {
				for (int n(0); n < kernelWidth; ++n) {
					for (int c(0); c < kernelDepth; ++c) {
						file << Q.layer(l).kernels[k].weights(m, n, c) << "\n";
					}
				}
			}

			file << Q.layer(l).kernels[k].bias << "\n";
		}
	}
}

void Agent::loadFromFile(const std::string& path)
{
	std::ifstream file;
	file.open(path);

	int kernelHeight(0), kernelWidth(0), kernelDepth(0);

	for (int l(0); l < Q.layers(); ++l) {
		kernelHeight = Q.layer(l).kernels[0].weights.height();
		kernelWidth = Q.layer(l).kernels[0].weights.width();
		kernelDepth = Q.layer(l).kernels[0].weights.depth();

		file >> Q.layer(l).stride;
		file >> Q.layer(l).padding;

		for (int k(0); k < Q.layer(l).kernels.size(); ++k) {
			for (int m(0); m < kernelHeight; ++m) {
				for (int n(0); n < kernelWidth; ++n) {
					for (int c(0); c < kernelDepth; ++c) {
						file >> Q.layer(l).kernels[k].weights(m, n, c);
					}
				}
			}

			file >> Q.layer(l).kernels[k].bias;
		}
	}
}

double Agent::_priority(double p, double alpha, double epsilon)
{
	return std::pow(std::abs(p) + epsilon, alpha);
}
