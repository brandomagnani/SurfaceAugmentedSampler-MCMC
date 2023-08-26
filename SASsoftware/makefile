# makefile

ModelDir = models/WarpedTorus3D


mainExec : main.o model.o SAS.o
	g++ -std=c++14 -lclapack main.o model.o SAS.o -o mainExec

model.o : $(ModelDir)/model.cpp $(ModelDir)/model.hpp
	g++ -std=c++14 -I$(ModelDir) -c $(ModelDir)/model.cpp

SAS.o : SAS.cpp SAS.hpp $(ModelDir)/model.hpp
	g++ -std=c++14 -I$(ModelDir) -c SAS.cpp

main.o : main.cpp $(ModelDir)/model.hpp SAS.hpp
	g++ -std=c++14 -I$(ModelDir) -c main.cpp

