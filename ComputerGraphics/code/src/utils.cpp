#include "utils.hpp"

std::random_device rd;
std::default_random_engine eng(rd());
std::uniform_real_distribution<double> distr(0.0, 1.0);

double INF = 2021012957.;

Vector3f vecInf = Vector3f(INF, INF, INF);

double rand01() { return distr(eng); }

double max(const Vector3f &v) {
    return (
        (v.x() > v.y() && v.x() > v.z()) ? v.x() :
        ((v.y() > v.z()) ? v.y() : v.z())
    );
}

double min(const Vector3f &v) {
    return (
        (v.x() < v.y() && v.x() < v.z()) ? v.x() :
        ((v.y() < v.z()) ? v.y() : v.z())
    );
}

Vector3f minVec(const Vector3f &v1, const Vector3f &v2) {
    return Vector3f(
        std::min(v1[0], v2[0]),
        std::min(v1[1], v2[1]),
        std::min(v1[2], v2[2])
    );
}

Vector3f maxVec(const Vector3f &v1, const Vector3f &v2) {
    return Vector3f(
        std::max(v1[0], v2[0]),
        std::max(v1[1], v2[1]),
        std::max(v1[2], v2[2])
    );
}