#ifndef HIT_H
#define HIT_H

#include <vecmath.h>
#include "ray.hpp"

class Material;

class Hit {
public:

    // constructors
    Hit() {
        material = nullptr;
        t = 1e38;
    }

    Hit(float _t, Material *m, const Vector3f &n) {
        t = _t;
        material = m;
        normal = n;
    }

    Hit(const Hit &h) {
        t = h.t;
        material = h.material;
        normal = h.normal;
    }

    // destructor
    ~Hit() = default;

    float getT() const {
        return t;
    }

    Material *getMaterial() const {
        return material;
    }

    const Vector3f &getNormal() const {
        return normal;
    }

    const bool getInto() const {
        return into;
    }

    const Vector2f getPos() const {
        return text_pos;
    }

    void set(float _t, Material *m, const Vector3f &n, bool _into = true, const Vector2f &pos = Vector2f()) {
        t = _t;
        material = m;
        normal = n;
        into = _into;
        text_pos = pos;
    }

private:
    float t;
    Material *material;
    Vector3f normal;
    bool into; /* 光线是否在内部 */
    Vector2f text_pos; /* 纹理坐标 */

};

inline std::ostream &operator<<(std::ostream &os, const Hit &h) {
    os << "Hit <" << h.getT() << ", " << h.getNormal() << ">";
    return os;
}

#endif // HIT_H
