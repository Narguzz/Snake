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

struct Node
{
	Node();

	double value;
	Transition* transition;
};

class ReplayMemory
{
public:
	ReplayMemory(size_t);
	~ReplayMemory();

	void push(Transition*, double);
	void setVal(size_t, double);

	std::vector<size_t> sample(size_t) const;
	
	const Node& operator[](size_t) const;

private:
	void _update(size_t, double);
	size_t _retrieve(size_t, double) const;

	size_t mFirstLeaf;
	size_t mPos;
	std::vector<Node> mNodes;
};

#endif // REPLAYMEMORY_H
