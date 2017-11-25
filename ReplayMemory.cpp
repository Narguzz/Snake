#include "ReplayMemory.h"

Transition::Transition()
{
}

Transition::Transition(Game& g, Direction a) :
	state({ g.state() }), action(a), reward(g.nextState(action)), nextState({ g.state() }), isTerminal(g.isFinished())
{
}

ReplayMemory::ReplayMemory(size_t capacity) :
	mCapacity(capacity),
	mPos(0)
{
}

size_t ReplayMemory::capacity() const
{
	return mCapacity;
}

std::vector<Transition> ReplayMemory::sample(size_t n) const
{
	if (n > mMemory.size())
		return mMemory;

	std::vector<Transition> batch(n);
	std::unordered_set<size_t> set;

	std::mt19937 generator(std::random_device{}());
	std::uniform_int_distribution<size_t> distribution(0, mMemory.size() - 1);

	while (set.size() < n)
		set.emplace(distribution(generator));

	size_t i(0);

	for (size_t x : set) {
		batch[i] = mMemory[x];
		++i;
	}

	return batch;
}

void ReplayMemory::push(const Transition& t)
{
	if (mPos >= mMemory.size())
		mMemory.push_back(t);
	else
		mMemory[mPos] = t;
	
	++mPos %= capacity();
}
