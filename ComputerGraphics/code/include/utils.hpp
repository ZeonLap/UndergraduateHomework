#ifndef UTILS_H
#define UTILS_H

#include <random>
#include <vecmath.h>

extern double INF;

extern Vector3f vecInf;

double rand01();

double max(const Vector3f &v);

double min(const Vector3f &v);

Vector3f minVec(const Vector3f &v1, const Vector3f &v2);

Vector3f maxVec(const Vector3f &v1, const Vector3f &v2);

#endif