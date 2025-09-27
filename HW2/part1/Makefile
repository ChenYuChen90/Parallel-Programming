CXX := g++
CXXFLAGS := -O3 -std=c++17 -Wall -march=native -pthread

all: pi.out

pi.o: pi.cpp
	$(CXX) $(CXXFLAGS) -c pi.cpp

pi.out: pi.o
	$(CXX) pi.o -o pi.out

clean:
	$(RM) *.o pi.out *~
