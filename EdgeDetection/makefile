canny: canny_main.o canny.o point.o
	g++ -o canny.out canny_main.o canny.o point.o
canny_main.o: canny_main.cpp
	g++ -c canny_main.cpp
canny.o: canny.cpp
	g++ -c canny.cpp
point.o: point.cpp
	g++ -c point.cpp
clean:
	rm -f *.o *.out output_images/*.pgm
