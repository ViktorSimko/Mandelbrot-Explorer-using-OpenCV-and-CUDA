#ifndef FRACTAL_H
#define FRACTAL_H

#include <cstdio>

class Fractal
{
public:
    Fractal(double cx0, double cy0, double cx1, double cy1, int width, int height, int maxIter);
    ~Fractal();
    void SetDimensions(double cx0, double cy0, double cx1, double cy1, int maxIter);
    char *GetImageBuffer();
private:
    double cx0, cy0, cx1, cy1;
    int width, height, maxIter;
    char *imageBuffer;
};

#endif // END FRACTAL_H
