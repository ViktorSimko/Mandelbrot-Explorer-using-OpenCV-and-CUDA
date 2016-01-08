#include "complexnumber.hpp"

ComplexNumberFr::ComplexNumberFr(double real, double imag)
{
    this->real = real; this->imag = imag;
}

ComplexNumberFr ComplexNumberFr::Square()
{
    this->real = this->real * this->real - this->imag * this->imag;
    this->imag = 2.0 * this->real * this->imag;
    return this;
}

double ComplexNumberFr::Abs()
{
    return sqrt(this->x * this->x + this->y * this->y);
}

void ComplexNumberFr::operator + (const ComplexNumberFr& numToAdd)
{
    this->real += numToAdd.GetReal();
    this->imag += numToAdd.GetImag();
}

void ComplexNumberFr::operator = (const ComplexNumberFr& numToAdd)
{
    this->real = numToAdd.GetReal();
    this->imag = numToAdd.GetImag();
}
