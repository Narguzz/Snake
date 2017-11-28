#include "Game.h"

Game::Game(int gridSize) :
	mGenerator(std::random_device{}()),
	mRandCoord(0, gridSize - 1),
	mGrid(2, Eigen::Matrix<bool, -1, -1>(gridSize, gridSize))
{
	initialize();
}

void Game::initialize()
{
	for (size_t i(0); i < mGrid[0].rows(); ++i)
		for (size_t j(0); j < mGrid[0].cols(); ++j)
			mGrid[0](i, j) = mGrid[1](i, j) = false; // Empty grid

	mBody = { { mRandCoord(mGenerator), mRandCoord(mGenerator) } };
	mGrid[0](mBody[0].x, mBody[0].y) = true;
	mDirectionList = { Right };
	mNbApples = 0;
	mScore = 0.;

	mIsFinished = false;

	_generateApple();
}

double Game::nextState(Direction move)
{
	if (move == None) return 0.0;

	double oldScore(mScore);

	// If the snake dies
	if (!_moveSnake(move)) {
		mIsFinished = true;
		return -1.0;
	}

	return mScore - oldScore; // = reward
}

bool Game::isFinished() const
{
	return mIsFinished;
}

Tensor3D Game::state() const
{
	Tensor3D state(mGrid[0].rows(), mGrid[0].cols(), 1);

	for (size_t i(0); i < mGrid[0].rows(); ++i) {
		for (size_t j(0); j < mGrid[0].cols(); ++j) {
			int x = (i - mBody[0].x + int(mGrid[0].rows() * 1.5)) % mGrid[0].rows(),
				y = (j - mBody[0].y + int(mGrid[0].cols() * 1.5)) % mGrid[0].cols();

			
			state(x, y, 0) = std::min(mGrid[0](i, j) + mGrid[1](i, j) * 0.299, 1.0);
		}
	}

	return state;
}

std::vector<Eigen::Matrix<bool, -1, -1>> Game::grid() const
{
	return mGrid;
}

double Game::score() const
{
	return mScore;
}

void Game::_generateApple()
{
	mGrid[1](mApple.x, mApple.y) = false;

	do {
		mApple = { mRandCoord(mGenerator), mRandCoord(mGenerator) };
	} while (mGrid[0](mApple.x, mApple.y));

	mGrid[1](mApple.x, mApple.y) = true;
}

bool Game::_moveSnake(Direction nextDir)
{
	if (isFinished())
		return false;

	std::list<Direction>::iterator it(mDirectionList.begin());
	Coords last(mBody.back());

	for (size_t i(0); i < mBody.size(); ++i, ++it) {
		mGrid[0](mBody[i].x, mBody[i].y) = false;

		switch (*it) {
		case Up:
			mBody[i].x += mGrid[0].rows() - 1;
			break;

		case Down:
			mBody[i].x += 1;
			break;

		case Left:
			mBody[i].y += mGrid[0].cols() - 1;
			break;

		case Right:
			mBody[i].y += 1;
			break;

		}

		mBody[i].x %= mGrid[0].rows();
		mBody[i].y %= mGrid[0].cols();
	}

	// If the snakes eats an apple...
	if (mNbApples) {
		mBody.push_back(last);
		--mNbApples;

		_generateApple();
	} else {
		mDirectionList.pop_back();
	}

	mDirectionList.push_front(nextDir);


	for (Coords pos : mBody) {
		// If the snake eats itself, he dies
		if (mGrid[0](pos.x, pos.y))
			return false;

		if (mGrid[1](pos.x, pos.y)) {
			++mNbApples;
			mScore += 1.0;
		}

		mGrid[0](pos.x, pos.y) = true;
	}

	// The snake lives !
	return true;
}
