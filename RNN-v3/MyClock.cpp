#include "MyClock.h"



MyClock::MyClock(string title)
{
	this->t_begin = clock();
	this->title = title;
}


MyClock::~MyClock()
{
}

void MyClock::showDuration()
{
	this->t_end = clock();
	cout << "========== "<< title <<" =============\ntime: "
		<< (double)(t_end - t_begin) / CLOCKS_PER_SEC
		<< " s" << endl;

}
