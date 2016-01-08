#ifndef COMPLEXNUMBER_H
#define COMPLEXNUMBER_H

#include <cmath>

class ComplexNumberFr
{
public:
    __host__ __device__ ComplexNumberFr(double real, double imag);
    __host__ __device__ virtual ~ComplexNumberFr();
    __host__ __device__ inline double GetReal(){ return this->real; }
    __host__ __device__ inline double GetImag(){ return this->imag; }
    __host__ __device__ ComplexNumberFr Square();
    __host__ __device__ double Abs();
    __host__ __device__ void operator + (const ComplexNumberFr& numToAdd);
    __host__ __device__ void operator = (const ComplexNumberFr& numToBeEqual);
private:
    double real;
    double imag;
};

#endif // END COMPLEXNUMBER_H
