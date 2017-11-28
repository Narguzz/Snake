#ifndef GAME_H
#define GAME_H

#include <list>
#include <random>
#include <iostream>
#include "Tensor3D.h"

enum Direction {
	Up,
	Right,
	Down,
	Left,
	None
};

struct Coords
{
	int x, y;
};

class Game
{
public:
	Game(int);
	void initialize();
	double nextState(Direction);

	bool isFinished() const;

	Tensor3D state() const;
	std::vector<Eigen::Matrix<bool, -1, -1>> grid() const;
	double score() const;

private:
	void _generateApple();
	bool _moveSnake(Direction);

	std::vector<Eigen::Matrix<bool, -1, -1>> mGrid;

	std::vector<Coords> mBody;
	std::list<Direction> mDirectionList;
	
	Coords mApple;
	size_t mNbApples;

	double mScore;

	std::mt19937 mGenerator;
	std::uniform_int_distribution<int> mRandCoord;

	bool mIsFinished;
};

#endif // GAME_H