#pragma once
#include <time.h>
#include <iostream>
#include <string>

using namespace std;

/*
	my clock.
	stop the clock, it'll print out the duration in a nice format.
*/
class MyClock
{
public:
	MyClock(string title);
	~MyClock();

	/*
		stop the timer and print the duration
	*/
	void stopAndShow();

private:
	clock_t t_begin;
	clock_t t_end;
	string title;

};

