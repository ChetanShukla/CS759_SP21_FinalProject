canny: canny_main.o canny.o hashmap.o
	g++ -o canny.out canny_main.o canny.o hashmap.o
canny_main.o: canny_main.cpp
	g++ -c canny_main.cpp
canny.o: canny.cpp
	g++ -c canny.cpp
hashmap.o: hashmap.cpp
	g++ -c hashmap.cpp
clean:
	rm -f *.o *.out output_images/*.pgm
