#include "fractal.hpp"

#ifdef __CUDACC__

__global__ static void calculateMandelbrot(char *imageBuffer, double cx0, double cy0, double cx1, double cy1,
                                           int width, int height, int maxIter);

#define cudaCheck(ins) { _cudaCheck(ins, __FILE__, __LINE__); }

inline void _cudaCheck(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "cudaCheck: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

#else

static void calculateMandelbrotCPU(char *imageBuffer, double cx0, double cy0, double cx1, double cy1,
                                   int width, int height, int maxIter);

#endif

Fractal::Fractal(double cx0, double cy0, double cx1, double cy1, int width, int height, int maxIter)
{
    this->cx0 = cx0;
    this->cy0 = cy0;
    this->cx1 = cx1;
    this->cy1 = cy1;
    this->width = width;
    this->height = height;
    this->maxIter = maxIter;
    this->imageBuffer = new char[width * height * 3];
}

Fractal::~Fractal()
{
    delete[] this->imageBuffer;
}

void Fractal::SetDimensions(double cx0, double cy0, double cx1, double cy1, int maxIter)
{
    this->cx0 = cx0;
    this->cy0 = cy0;
    this->cx1 = cx1;
    this->cy1 = cy1;
    this->maxIter = maxIter;
}

char *Fractal::GetImageBuffer()
{

#ifdef __CUDACC__

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(this->width / threadsPerBlock.x, this->height / threadsPerBlock.y);

    char *imageBuffer_d;

    cudaCheck(cudaMalloc(&imageBuffer_d, width * height * 3));
    calculateMandelbrot<<<blocksPerGrid, threadsPerBlock>>>(imageBuffer_d, this->cx0, this->cy0, this->cx1, this->cy1,
                                                            this->width, this->height, this->maxIter);

    cudaCheck(cudaMemcpy(imageBuffer, imageBuffer_d, width * height * 3, cudaMemcpyDeviceToHost));
    cudaCheck(cudaFree(imageBuffer_d));

#else

    calculateMandelbrotCPU(this->imageBuffer, this->cx0, this->cy0, this->cx1, this->cy1,
                           this->width, this->height, this->maxIter);

#endif

    return this->imageBuffer;
}

#ifdef __CUDACC__

__global__ static void calculateMandelbrot(char *imageBuffer, double cx0, double cy0, double cx1, double cy1,
                                           int width, int height, int maxIter)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int pixelId = (row * width + col) * 3;

    double x = 0, y = 0;
    double cx = (double)col / width * (cx1 - cx0) + cx0;
    double cy = (double)row / height * (cy0 - cy1) + cy1;

    int numberOfIterations = 0;
    double tempx;

    while((x * x + y * y  < 4.0) && (numberOfIterations <= maxIter))
    {
        tempx = x * x - y * y + cx;
        y = 2.0 * x * y + cy;
        x = tempx;
        numberOfIterations++;
    }
    
    int color = numberOfIterations;

    if (numberOfIterations == maxIter) color = 0;

    imageBuffer[pixelId] = 255 - color % 256;//color % 256;
    imageBuffer[pixelId + 1] = 0;
    imageBuffer[pixelId + 2] = color * 5 % 256;
}

#else

static void calculateMandelbrotCPU(char *imageBuffer, double cx0, double cy0, double cx1, double cy1,
                                   int width, int height, int maxIter)
{

    for(int j = 0; j < width; j++)
    for(int k = 0; k < height; k++)
    {
        int row = k;
        int col = j;
        int pixelId = (row * width + col) * 3;

        double x = 0, y = 0;
        double cx = (double)col / width * (cx1 - cx0) + cx0;
        double cy = (double)row / height * (cy0 - cy1) + cy1;

        int numberOfIterations = 0;
        double tempx;

        while((x * x + y * y  < 4.0) && (numberOfIterations <= maxIter))
        {
            tempx = x * x - y * y + cx;
            y = 2.0 * x * y + cy;
            x = tempx;
            numberOfIterations++;
        }
        
        int color = numberOfIterations;

        if (numberOfIterations == maxIter) color = 0;

        imageBuffer[pixelId] = 255 - color % 256;//color % 256;
        imageBuffer[pixelId + 1] = 0;
        imageBuffer[pixelId + 2] = color * 5 % 256;
    }
}

#endif



