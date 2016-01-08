#include "fractal.hpp"
#include <iostream>
#include <cstring>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

int width = 1024, height = 1024, maxIter = 250;
double cx0 = -2, cy0 = 1.5, cx1 = 1, cy1 = -1.5;

bool clicked = false;
bool isImgCorrect = false;

void reset()
{
    cx0 = -2; cy0 = 1.5; cx1 = 1; cy1 = -1.5;
    maxIter = 250;
    isImgCorrect = false;
}

void mouseCallBack(int event, int x, int y, int flags, void *userdata);

int main(int argc, char *argv[])
{
    Mat output(width, height, CV_8UC3);

    Fractal mandelbrot(cx0, cy0, cx1, cy1, width, height, maxIter);

    namedWindow("MandelWindow", 1);

    setMouseCallback("MandelWindow", mouseCallBack);

    for(;;)
    {
        if (!isImgCorrect)
        {
            mandelbrot.SetDimensions(cx0, cy0, cx1, cy1, maxIter);
            memcpy(output.data, mandelbrot.GetImageBuffer(), width * height * 3);

            imshow("MandelWindow", output);

            isImgCorrect = true;
        }
        int key = waitKey(5);

        switch(key)
        {
        case 'q':
            return 0;
        case 'z':
            cx0 *= (double)2 / 3;
            cy0 *= (double)2 / 3;
            cx1 *= (double)2 / 3;
            cy1 *= (double)2 / 3;
            cout << "z pressed" << endl;
            isImgCorrect = false;
            break;
        case 'i':
            maxIter *= 2;
            cout << maxIter << endl;
            isImgCorrect = false;
            break;
        case 'r':
            reset();
            break;
        }
    }
    return 0;
}

double tempcx0, tempcy0, tempcx1, tempcy1;

void mouseCallBack(int event, int x, int y, int flags, void *userdata)
{
    if(event == EVENT_LBUTTONDOWN)
    {
        cout << "LButtonDown x: " << x << "\ty: " << y << endl;
        tempcx0 = tempcy0 = tempcx1 = tempcy1 = 0;
        tempcx0 = (double)x / width * (cx1 - cx0) + cx0;
        tempcy0 = (double)y / height * (cy0 - cy1) + cy1;
        clicked = true;
    }

    if(event == EVENT_LBUTTONUP)
    {
        cout << "LButtonUp x: " << x << "\ty: " << y << endl;
        tempcx1 = (double)x / width * (cx1 - cx0) + cx0;
        tempcy1 = tempcy0 - (tempcx1 - tempcx0);
        cx0 = tempcx0; cx1 = tempcx1;
        cy0 = tempcy0; cy1 = tempcy1;
        isImgCorrect = false;
        clicked = false;
    }

    if(event == EVENT_MOUSEMOVE && clicked)
    {
        cout << "MouseMove x: " << x << "\ty: " << y << endl;
    }
}
