#ifndef REPLAYMEMORY_H
#define REPLAYMEMORY_H

#include <numeric>
#include <unordered_set>
#include "Game.h"

struct Transition
{
	Transition();
	Transition(Game&, Direction);

	Tensor3D state;
	Direction action;
	Tensor3D nextState;
	double reward;
	bool isTerminal;
};


// = Circular buffer
class ReplayMemory
{
public:
	ReplayMemory(size_t);

	size_t capacity() const;
	std::vector<Transition> sample(size_t) const;

	void push(const Transition&);

private:
	size_t mCapacity, mPos;
	std::vector<Transition> mMemory;
};

#endif // REPLAYMEMORY_H
