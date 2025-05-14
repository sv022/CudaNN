#pragma once
#include <iostream>
#include <vector>
#include <stdio.h>
#include <string>
#include<chrono>

class ProgressBar
{
public:
	ProgressBar(char notDoneChar, char doneChar, unsigned int size);
	void end();
	
	unsigned int todo;
	unsigned int done;
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	
	void fillUpCells(unsigned int cells);
	void fillUp();
	void displayPercentage();
	void displayTasksDone();
	void displayTimeElapsed();

private:
	unsigned int size = 0;
	unsigned int pos = 1;
	char c;
	char ch;
	std::vector <char> bar;
};


ProgressBar::ProgressBar(char notDoneChar, char doneChar, unsigned int size)
:c(doneChar), ch(notDoneChar), size(size), todo(0), done(0) {
	if(size <= 100)
		size = size;	
	else
		size = 100;

	bar.push_back('[');

	for(int i = 1; i < size + 1; i++) bar.push_back(ch);

	bar.push_back(']');
}

void ProgressBar::fillUpCells(unsigned int cells) {
	pos = 0;
	for(int i = 1; i < cells; i++) {
		bar[i] = c;
		std::cout << '\r';

		for(int j = 0; j < bar.size(); j++) std::cout << bar[j] << std::flush;
	}
	pos += cells;
}

void ProgressBar::fillUp() {
	bar[pos] = c;
	pos++;
	
	std::cout << '\r';
	
	for (int i = 0; i < bar.size(); i++) {
		std::cout << bar[i] << std::flush;
	}
}

void ProgressBar::displayPercentage() {
	float percent = ((float)pos / (float)(bar.size() - 1)) * 100;
	std::cout << (int)percent << "%";
}

void ProgressBar::displayTasksDone() {
	std::cout << '(' << done << '/' << todo << ')' << std::flush;
}

void ProgressBar::displayTimeElapsed() {
	std::cout << '(' << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin).count() / 1000 << "s)" << std::flush;
}

void ProgressBar::end() {
	std::cout << std::endl;
}