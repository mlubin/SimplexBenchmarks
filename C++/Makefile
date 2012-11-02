all: runbench runbenchBoundsCheck

runbench: runbench.cpp common.cpp
	g++ -O2 -std=c++0x -o runbench runbench.cpp

runbenchBoundsCheck: runbenchBoundsCheck.cpp common.cpp
	g++ -O2 -std=c++0x -o runbenchBoundsCheck runbenchBoundsCheck.cpp
