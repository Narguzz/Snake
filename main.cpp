#include "Agent.h"

int main()
{
	Agent agent;
	agent.train(-1, 16, 262144, 0.99, 1.0, 0.01, 0.0005, 0.00025, 0.95, 1e-8);

	return 0;
}
