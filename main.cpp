#include "Agent.h"

int main()
{
	Agent agent;
	agent.saveToFile("weights.txt");

	agent.train(-1, 16, 131072, 0.99, 1.0, 0.05, 0.000001, 0.95, 1e-6);

	system("PAUSE");

	return 0;
}
