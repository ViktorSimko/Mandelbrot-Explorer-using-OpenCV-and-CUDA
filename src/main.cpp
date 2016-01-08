#include "fractal.hpp"
#include <iostream>
#include <cmath>
#include <cstring>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

int width = 1024, height = 1024, maxIter = 250;
double cx0 = -2, cy0 = 1.5, cx1 = 1, cy1 = -1.5;

Mat output(width, height, CV_8UC3);

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
int rx0, ry0, rx1, ry1;

void mouseCallBack(int event, int x, int y, int flags, void *userdata)
{
    if(event == EVENT_LBUTTONDOWN)
    {
        //cout << "LButtonDown x: " << x << "\ty: " << y << endl;
        rx0 = x; ry0 = y;
        tempcx0 = tempcy0 = tempcx1 = tempcy1 = 0;
        clicked = true;
    }

    if(event == EVENT_LBUTTONUP)
    {
        //cout << "LButtonUp x: " << x << "\ty: " << y << endl;
        tempcx0 = (double)rx0 / width * abs(cx0 - cx1) + min(cx0, cx1);
        tempcy0 = (double)ry0 / height * abs(cy0 - cy1) + min(cy0, cy1);
        tempcx1 = (double)rx1 / width * abs(cx0 - cx1) + min(cx0, cx1);
        tempcy1 = (double)ry1 / height * abs(cy0 - cy1) + min(cy0, cy1);
        cx0 = tempcx0; cx1 = tempcx1;
        cy0 = tempcy0; cy1 = tempcy1;
        cout << cx0 << ' ' << cy0 << ' ' << cx1 << ' ' << cy1 << endl;
        isImgCorrect = false;
        clicked = false;
    }

    if(event == EVENT_MOUSEMOVE && clicked)
    {
        //cout << "MouseMove x: " << x << "\ty: " << y << endl;
        Mat outputTemp = output.clone();
        
        rx1 = x; ry1 = ry0 + (((ry0 - y) < 0) ? 1 : -1) * abs(rx0 - x);
        rectangle(outputTemp, Point(rx0, ry0), Point(rx1, ry1), Scalar(255, 255, 255));

        imshow("MandelWindow", outputTemp);
    }
}
