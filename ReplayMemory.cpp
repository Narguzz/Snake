#include "ReplayMemory.h"

Transition::Transition()
{
}

Transition::Transition(Game& g, Direction a)
{
	state = g.state();
	action = a;
	reward = g.nextState(a);
	nextState = g.state();
	isTerminal = g.isFinished();
}


Node::Node() : transition(nullptr), value(0.0)
{
}

ReplayMemory::ReplayMemory(size_t leaves)
{
	size_t size = std::pow(2, std::ceil(std::log2(leaves))) + leaves - 1;
	mNodes = std::vector<Node>(size);
	mFirstLeaf = size - leaves;
	
	mPos = mFirstLeaf;
}

ReplayMemory::~ReplayMemory()
{
	for (Node& node : mNodes)
		delete node.transition;
}

void ReplayMemory::push(Transition* t, double priority)
{
	delete mNodes[mPos].transition;
	mNodes[mPos].transition = t;
	setVal(mPos - mFirstLeaf, priority);

	++mPos;

	if (mPos >= mNodes.size())
		mPos = mFirstLeaf;
}

// The user can only update leaves
void ReplayMemory::setVal(size_t k, double newVal)
{
	size_t i(mFirstLeaf + k % (mNodes.size() - mFirstLeaf));
	_update(i, newVal - mNodes[i].value);
}

std::vector<size_t> ReplayMemory::sample(size_t n) const
{
	std::vector<size_t> batch(n);

	std::mt19937 generator(std::random_device{}());
	std::uniform_real_distribution<double> distribution(0.0, mNodes[0].value);

	for (size_t& k : batch)
		k = _retrieve(0, distribution(generator));

	return batch;
}

// Access a leaf node
const Node & ReplayMemory::operator[](size_t k) const
{
	return mNodes[mFirstLeaf + k % (mNodes.size() - mFirstLeaf)];
}

// Recursive function
void ReplayMemory::_update(size_t k, double delta)
{
	mNodes[k].value += delta;

	// If we are not updating the root node, update parent (sum heap)
	if (k)
		_update((k - 1) / 2, delta);
}

size_t ReplayMemory::_retrieve(size_t n, double sum) const
{
	if (n >= mFirstLeaf)
		return n - mFirstLeaf;

	if (mNodes[2 * n + 1].value >= sum)
		return _retrieve(2 * n + 1, sum); // 2n + 1 is the index of the left child of the node at index n
	else
		return _retrieve(2 * n + 2, sum - mNodes[2 * n + 1].value);
}
