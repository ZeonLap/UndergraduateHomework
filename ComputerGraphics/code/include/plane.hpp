#ifndef PLANE_H
#define PLANE_H

#include "object3d.hpp"
#include <vecmath.h>
#include <cmath>

class Plane : public Object3D {
public:
    Plane() = delete;

    Plane(const Vector3f &normal, float d, Material *m) : Object3D(m), normal(normal.normalized()), d(d) {
        
    }

    ~Plane() override = default;


    // 采用循环的材质
    Vector2f getPos(const Vector3f &m_pos) {
        Vector2f pos;
        if (fabs(normal.y()) > .1) { pos = m_pos.xz(); }
        else if (fabs(normal.x() > .1)) { pos = m_pos.yz(); }
        else { pos = m_pos.xy(); }
        return pos;
    }

    bool intersect(const Ray &r, Hit &h, float tmin) override {
        Vector3f R0 = r.getOrigin(), Rd = r.getDirection();
        float t = - ( - d + Vector3f::dot(normal, R0)) / (Vector3f::dot(normal, Rd));
        if (t >= 0 && t > tmin && t < h.getT()) {
            Vector3f N = Vector3f::dot(normal, Rd) > 0 ? -normal : normal;
            Vector3f m_pos = r.pointAtParameter(t);
            Vector2f pos = getPos(m_pos);
            h.set(t, material, N, true, pos / 90.0);
            return true;
        }
        else {
            return false;
        }
    }

protected:
    Vector3f normal;
    float d;

};

#endif //PLANE_H
		

