#include "Agent.h"

int main()
{
	Agent agent;
	agent.saveToFile("weights.txt");

	agent.train(-1, 16, 2097152, 0.9, 1.0, 0.05, 0.000001, 0.95, 1e-6);

	system("PAUSE");

	return 0;
}