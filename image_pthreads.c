#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <pthread.h>
#include "image.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define DEFAULT_THREADS 4

//An array of kernel matrices to be used for image convolution.  
//The indexes of these match the enumeration from the header file. ie. algorithms[BLUR] returns the kernel corresponding to a box blur.
Matrix algorithms[]={
    {{0,-1,0},{-1,4,-1},{0,-1,0}},
    {{0,-1,0},{-1,5,-1},{0,-1,0}},
    {{1/9.0,1/9.0,1/9.0},{1/9.0,1/9.0,1/9.0},{1/9.0,1/9.0,1/9.0}},
    {{1.0/16,1.0/8,1.0/16},{1.0/8,1.0/4,1.0/8},{1.0/16,1.0/8,1.0/16}},
    {{-2,-1,0},{-1,1,1},{0,1,2}},
    {{0,0,0},{0,1,0},{0,0,0}}
};

typedef struct {
    Image *srcImage;
    Image *destImage;
    Matrix algorithm;
    int startRow;
    int endRow;
} ThreadData;

//getPixelValue - Computes the value of a specific pixel on a specific channel using the selected convolution kernel
//Parameters: srcImage:  An Image struct populated with the image being convoluted
//            x: The x coordinate of the pixel
//            y: The y coordinate of the pixel
//            bit: The color channel being manipulated
//            algorithm: The 3x3 kernel matrix to use for the convolution
//Returns: The new value for this x,y pixel and bit channel
uint8_t getPixelValue(Image* srcImage, int x, int y, int bit, Matrix algorithm) {
    int px, mx, py, my, span;
    span = srcImage->width * srcImage->bpp;
    // for the edge pixels, just reuse the edge pixel
    px = x+1; py = y+1; mx = x-1; my = y-1;
    if (mx < 0) mx = 0;
    if (my < 0) my = 0;
    if (px >= srcImage->width)  px = srcImage->width  - 1;
    if (py >= srcImage->height) py = srcImage->height - 1;

    // Use int to avoid uint8_t overflow/underflow during accumulation.
    // Kernels like edge detection produce negative sums; storing those
    // directly into uint8_t wraps around and corrupts the image (grey output).
    int result =
        algorithm[0][0] * srcImage->data[Index(mx, my, srcImage->width, bit, srcImage->bpp)] +
        algorithm[0][1] * srcImage->data[Index(x,  my, srcImage->width, bit, srcImage->bpp)] +
        algorithm[0][2] * srcImage->data[Index(px, my, srcImage->width, bit, srcImage->bpp)] +
        algorithm[1][0] * srcImage->data[Index(mx, y,  srcImage->width, bit, srcImage->bpp)] +
        algorithm[1][1] * srcImage->data[Index(x,  y,  srcImage->width, bit, srcImage->bpp)] +
        algorithm[1][2] * srcImage->data[Index(px, y,  srcImage->width, bit, srcImage->bpp)] +
        algorithm[2][0] * srcImage->data[Index(mx, py, srcImage->width, bit, srcImage->bpp)] +
        algorithm[2][1] * srcImage->data[Index(x,  py, srcImage->width, bit, srcImage->bpp)] +
        algorithm[2][2] * srcImage->data[Index(px, py, srcImage->width, bit, srcImage->bpp)];

    // Clamp to valid pixel range before casting back to uint8_t
    if (result < 0)   result = 0;
    if (result > 255) result = 255;
    return (uint8_t)result;
}

void *threadConvolute(void *arg) {
    ThreadData *td = (ThreadData *)arg;

    for (int row = td->startRow; row < td->endRow; row++) {
        for (int pix = 0; pix < td->srcImage->width; pix++) {
            for (int bit = 0; bit < td->srcImage->bpp; bit++) {
                td->destImage->data[Index(pix, row, td->srcImage->width, bit, td->srcImage->bpp)] =
                    getPixelValue(td->srcImage, pix, row, bit, td->algorithm);
            }
        }
    }

    return NULL;
}

//convolute:  Applies a kernel matrix to an image using pthreads.
//            Each thread handles an exclusive band of rows — no two threads
//            write the same index in destImage, so there are no race conditions.
//Parameters: srcImage: The image being convoluted
//            destImage: A pointer to a pre-allocated structure to receive the result
//            algorithm: The kernel matrix to use for the convolution
//            numThreads: Number of pthreads to spawn
//Returns: Nothing
void convolute(Image *srcImage, Image *destImage, Matrix algorithm, int numThreads) {
    if (numThreads < 1) numThreads = 1;
    if (numThreads > srcImage->height) numThreads = srcImage->height;

    pthread_t  *threads    = malloc(sizeof(pthread_t)  * numThreads);
    ThreadData *threadData = malloc(sizeof(ThreadData) * numThreads);

    int rowsPerThread = srcImage->height / numThreads;
    int extraRows     = srcImage->height % numThreads;
    int currentRow    = 0;

    for (int i = 0; i < numThreads; i++) {
        int myRows = rowsPerThread + (i < extraRows ? 1 : 0);

        threadData[i].srcImage  = srcImage;
        threadData[i].destImage = destImage;
        threadData[i].startRow  = currentRow;
        threadData[i].endRow    = currentRow + myRows;

        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
                threadData[i].algorithm[r][c] = algorithm[r][c];

        pthread_create(&threads[i], NULL, threadConvolute, &threadData[i]);
        currentRow += myRows;
    }

    for (int i = 0; i < numThreads; i++)
        pthread_join(threads[i], NULL);

    free(threads);
    free(threadData);
}

//Usage: Prints usage information for the program
//Returns: -1
int Usage() {
    printf("Usage: ./image_pthreads <filename> <type> [threads]\n");
    printf("\twhere type is one of (edge,sharpen,blur,gauss,emboss,identity)\n");
    return -1;
}

//GetKernelType: Converts the string name of a convolution into a value from the KernelTypes enumeration
//Parameters: type: A string representation of the type
//Returns: an appropriate entry from the KernelTypes enumeration, defaults to IDENTITY
enum KernelTypes GetKernelType(char* type) {
    if      (!strcmp(type, "edge"))    return EDGE;
    else if (!strcmp(type, "sharpen")) return SHARPEN;
    else if (!strcmp(type, "blur"))    return BLUR;
    else if (!strcmp(type, "gauss"))   return GAUSE_BLUR;
    else if (!strcmp(type, "emboss"))  return EMBOSS;
    else                               return IDENTITY;
}

//main:
//argv is expected to take 2-3 arguments. First is the source file name, second is the algorithm, third (optional) is thread count.
int main(int argc, char** argv) {
    long t1, t2;
    t1 = time(NULL);

    stbi_set_flip_vertically_on_load(0);
    if (argc < 3 || argc > 4) return Usage();

    char* fileName = argv[1];
    if (!strcmp(argv[1], "pic4.jpg") && !strcmp(argv[2], "gauss")) {
        printf("You have applied a gaussian filter to Gauss which has caused a tear in the time-space continum.\n");
    }

    enum KernelTypes type = GetKernelType(argv[2]);

    Image srcImage, destImage;
    srcImage.data = stbi_load(fileName, &srcImage.width, &srcImage.height, &srcImage.bpp, 0);
    if (!srcImage.data) {
        printf("Error loading file %s.\n", fileName);
        return -1;
    }

    destImage.bpp    = srcImage.bpp;
    destImage.height = srcImage.height;
    destImage.width  = srcImage.width;
    destImage.data   = malloc(sizeof(uint8_t) * destImage.width * destImage.bpp * destImage.height);

    int numThreads = (argc == 4) ? atoi(argv[3]) : DEFAULT_THREADS;
    convolute(&srcImage, &destImage, algorithms[type], numThreads);

    stbi_write_png("output.png", destImage.width, destImage.height, destImage.bpp,
                   destImage.data, destImage.bpp * destImage.width);

    stbi_image_free(srcImage.data);
    free(destImage.data);

    t2 = time(NULL);
    printf("Took %ld seconds with pthreads using %d thread(s)\n", (long)(t2 - t1), numThreads);
    return 0;
}
