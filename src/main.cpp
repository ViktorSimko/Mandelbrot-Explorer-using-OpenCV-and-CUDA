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
double cx0 = -2, cy0 = -1.5, cx1 = 1, cy1 = 1.5, rangex, rangey;

Mat output(width, height, CV_8UC3);

bool clicked = false;
bool isImgCorrect = false;

double tempcx0, tempcy0, tempcx1, tempcy1;

void reset()
{
    cx0 = -2; cy0 = -1.5; cx1 = 1; cy1 = 1.5;
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
        case 'k':
            rangey = cy1 - cy0;
            cy0 += ((double)1/5) * rangey;
            cy1 += ((double)1/5) * rangey;
            isImgCorrect = false;
            break;
        case 'j':
            rangey = cy1 - cy0;
            cy0 -= ((double)1/5) * rangey;
            cy1 -= ((double)1/5) * rangey;
            isImgCorrect = false;
            break;
        case 'l':
            rangex = cx1 - cx0;
            cx0 += ((double)1/5) * rangex;
            cx1 += ((double)1/5) * rangex;
            isImgCorrect = false;
            break;
        case 'h':
            rangex = cx1 - cx0;
            cx0 -= ((double)1/5) * rangex;
            cx1 -= ((double)1/5) * rangex;
            isImgCorrect = false;
            break;
        case 'z':
            rangex = cx1 - cx0;
            rangey = cy1 - cy0;
            tempcx0 = cx1 - ((double)4 / 5) * rangex;
            tempcy0 = cy1 - ((double)4 / 5) * rangey;
            cx1 = cx0 + ((double)4 / 5) * rangex;
            cy1 = cy0 + ((double)4 / 5) * rangey;
            cx0 = tempcx0;
            cy0 = tempcy0;
            //cout << "z pressed" << endl;
            isImgCorrect = false;
            break;
        case 'u':
            rangex = cx1 - cx0;
            rangey = cy1 - cy0;
            tempcx0 = cx1 - ((double)5 / 4) * rangex;
            tempcy0 = cy1 - ((double)5 / 4) * rangey;
            cx1 = cx0 + ((double)5 / 4) * rangex;
            cy1 = cy0 + ((double)5 / 4) * rangey;
            cx0 = tempcx0;
            cy0 = tempcy0;
            //cout << "z pressed" << endl;
            isImgCorrect = false;
            break;
        case 'i':
            maxIter *= 2;
            cout << "number of iterations: " << maxIter << endl;
            isImgCorrect = false;
            break;
        case 'd':
            maxIter /= 2;
            cout << "number of iterations: " << maxIter << endl;
            isImgCorrect = false;
            break;
        case 'r':
            reset();
            break;
        }
    }
    return 0;
}

int rx0, ry0, rx1, ry1;

void mouseCallBack(int event, int x, int y, int flags, void *userdata)
{
    if(event == EVENT_LBUTTONDOWN)
    {
        //cout << "LButtonDown x: " << x << "\ty: " << y << endl;
        rx0 = x; ry0 = y;
        clicked = true;
    }

    if(event == EVENT_LBUTTONUP)
    {
        //cout << "LButtonUp x: " << x << "\ty: " << y << endl;
        rangex = cx1 - cx0;
        rangey = cy1 - cy0;
        tempcx0 = (double)rx0 / width * rangex + cx0;
        tempcy0 = (double)(height - ry0) / height * rangey  + cy0;
        tempcx1 = (double)rx1 / width * rangex + cx0;
        tempcy1 = (double)(height - ry1) / height * rangey + cy0;
        cx0 = min(tempcx0, tempcx1); cx1 = max(tempcx0, tempcx1);
        cy0 = min(tempcy0, tempcy1); cy1 = max(tempcy0, tempcy1);
        //cout << cx0 << ' ' << cy0 << ' ' << cx1 << ' ' << cy1 << endl;
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
