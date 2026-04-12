all: image_pthreads image_openmp

image_pthreads: image_pthreads.c image.h
	gcc -g image_pthreads.c -o image_pthreads -lm -lpthread

image_openmp: image_openmp.c image.h
	gcc -g image_openmp.c -o image_openmp -lm -fopenmp

clean:
	rm -f image_pthreads image_openmp output.png