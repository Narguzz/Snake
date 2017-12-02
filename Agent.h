#ifndef AGENT_H
#define AGENT_H

#include "Network.h"
#include "Game.h"
#include "ReplayMemory.h"

#include <array>
#include <string>
#include <fstream>
#include <iostream>

class Agent
{
public:
	Agent();

	Direction optimalAction(const Tensor3D&, std::vector<Tensor3D>& = std::vector<Tensor3D>()) const;
	void train(size_t, size_t, size_t, double, double, double, double, double, double, double);

	void saveToFile(const std::string&) const;
	void loadFromFile(const std::string&);

private:
	double _priority(double, double, double);

	Network Q;
};

#endif // AGENT_H
