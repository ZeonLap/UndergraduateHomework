#ifndef AABB_H
#define AABB_H


#include <vecmath.h>
#include "ray.hpp"
#include "utils.hpp"

using std::min;
using std::max;

class AABB {
    /* "左下", "右上" */
    Vector3f ld, ru;

public:
    AABB() : ld(-vecInf), ru(vecInf) {

    }

    AABB(const Vector3f &_ld, const Vector3f &_ru) : ld(_ld), ru(_ru) {

    }

    void set(const Vector3f &_ld, const Vector3f &_ru) {
        ld = _ld;
        ru = _ru;
    }

    void reset() {
        ld = -vecInf;
        ru = vecInf;
    }

    void fit(const Vector3f &v) {
        set(minVec(ld, v), maxVec(ru, v));
    }

    Vector3f getld() const { return ld; }

    Vector3f getru() const { return ru; }


    bool intersect(const Ray &r) {
        if (ld == -vecInf && ru == vecInf) { return true; }
        Vector3f o = r.getOrigin(), d = r.getDirection();
        Vector3f rd(1. / d.x(), 1. / d.y(), 1. / d.z());
        Vector3f t1 = (ld - o) * rd, t2 = (ru - o) * rd;
        double t_enter = max(max(min(t1[0], t2[0]), min(t1[1], t2[1])), min(t1[2], t2[2]));
        double t_exit = min(min(max(t1[0], t2[0]), max(t1[1], t2[1])), max(t1[2], t2[2]));
        return (t_enter < t_exit) && (t_exit > 0);
    }

};


#endif