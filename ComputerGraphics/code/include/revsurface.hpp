#ifndef REVSURFACE_HPP
#define REVSURFACE_HPP

#include "object3d.hpp"
#include "curve.hpp"
#include "utils.hpp"
#include <tuple>
#include <cmath>

#define BEGIN       0.0
#define END         1.0
#define MAX_ITER    64
#define EPS         1e-4

using std::min;
using std::max;

class RevSurface : public Object3D {

    Curve *pCurve;

public:
    RevSurface(Curve *pCurve, Material* material) : pCurve(pCurve), Object3D(material) {
        // Check flat.
        Vector3f ld = -vecInf, ru = vecInf;
        for (const auto &cp : pCurve->getControls()) {
            if (cp.z() != 0.0) {
                printf("Profile of revSurface must be flat on xy plane.\n");
                exit(0);
            }
            Vector3f v(abs(cp.x()), cp.y(), abs(cp.z()));
            ld[0] = min(ld[0], -v[0]), ld[1] = min(ld[1], v[1]), ld[2] = min(ld[2], -v[0]);
            ru[0] = max(ru[0], v[0]), ru[1] = max(ru[1], v[1]), ru[2] = max(ru[2], v[0]);
        }
        this->setAABB(ld, ru);
    }

    ~RevSurface() override {
        delete pCurve;
    }

    bool intersect(const Ray &r, Hit &h, float tmin) override {
        // implement this for the ray-tracing routine using G-N iteration.
        if (!this->aabb.intersect(r)) return false;
        if (r.getDirection().y() != 0) { return Intersect(r, h, tmin); }
        else { return SpecialIntersect(r, h, tmin); }
    }

    bool SpecialIntersect(const Ray &r, Hit &h, double tmin) {
        bool res = false;
        const Vector3f &o = r.getOrigin(), &d = r.getDirection(), &rd = r.getRdirection();
        double d_x_y = d.x() * rd.y(), d_z_y = d.z() * rd.y();
        for (int i = 0; i < 128; ++i) {
            double t = BEGIN + (END - BEGIN) * (double)i / 128;
            for (int j = 0; j < MAX_ITER; ++j) {
                const Vector3f &point = pCurve->Point(t);
                const Vector3f &tangent = pCurve->Tangent(t);
                double fd = point.y() - o.y();
                if (fabs(fd) < EPS) {
                    double a = 1.0; // d.x() * d.x() + d.z() * d.z(), for d.y = 0
                    double half_b = d.x() * o.x() + d.z() * o.z();
                    double c = o.x() * o.x() + o.z() * o.z() - point.x() * point.x();
                    double r_t = (-half_b - sqrt(half_b * half_b - c)); // a = 1
                    if (tmin < r_t && r_t < h.getT()) {
                        double cos_theta = point.x() == 0 ? 0 : (r_t * d.x() + o.x()) / point.x();
                        double sin_theta = point.x() == 0 ? 1 : (r_t * d.z() + o.z()) / point.x();
                        Vector3f n = Vector3f::cross(t, -Vector3f::FORWARD);
                        Vector3f n_rot = Vector3f(
                            cos_theta * n.x() - sin_theta * n.z(),
                            n.y(),
                            sin_theta * n.x() + cos_theta * n.z()
                        );
                        bool into = true;
                        if (Vector3f::dot(n_rot, d) > 0) {
                            n_rot = -n_rot;
                            into = false;
                        }
                        h.set(r_t, material, n_rot, into);
                        res = true;
                    }
                    break;
                }
                double derive = tangent.y();
                t -= fd / (32 * derive);
                if (t < 0 || t > 1) break;
            }
        }
        return res;
    }

    bool Intersect(const Ray &r, Hit &h, float tmin) {
        // dy != 0
        // Equation: F(t) = (x_cos) ** 2 + (x_sin) ** 2 - x(t) ** 2 = 0
        // Tangent: F'(t) = 2 * (x_cos) * (dx / dy) * y'(t) + 
        //                  2 * (x_sin) * (dz / dy) * y'(t) -
        //                  2 * x(t) * x'(t)
        bool res = false;
        const Vector3f& o = r.getOrigin(), &d = r.getDirection(), &rd = r.getRdirection();
        for (int i = 0; i < 128; ++i) {
            double t = BEGIN + (END - BEGIN) * (double)i / 128;
            for (int j = 0; j < MAX_ITER; ++j) {
                const Vector3f &point = pCurve->Point(t);
                const Vector3f &tangent = pCurve->Tangent(t);
                double x_cos = (point.y() - o.y()) * d.x() * rd.y() + o.x(), x_sin = (point.y() - o.y()) * d.z() * rd.y() + o.z(), x = point.x();
                double fd = x_cos * x_cos + x_sin * x_sin - x * x;
                if (fabs(fd) < EPS) {
                    double tr = std::max((x_cos - o.x()) * rd.x(), (x_sin - o.z()) * rd.z());
                    if (tmin < tr && tr < h.getT()) {
                        double cos_theta = (d.x() * tr + o.x()) / x, sin_theta = (d.z() * tr + o.z()) / x;
                        Vector3f Pt(tangent.x() * cos_theta, tangent.y(), tangent.x() * sin_theta), Ptheta(-x * sin_theta, 0, x * cos_theta);
                        Vector3f normal = Vector3f::cross(Pt, Ptheta).normalized();
                        if (Vector3f::dot(normal, d) > 0) { normal = -normal; }
                        h.set(tr, material, normal);
                        res = true;
                        break;
                    }
                }
                double dif = 2 * (x_cos) * (d.x() / d.y()) * tangent.y() + 
                            2 * (x_sin) * (d.z() / d.y()) * tangent.y() - 
                            2 * x * tangent.x();
                t -= fd / dif;
                if (t < 0 || t > 1) { break; }
            }
        }
        return res;
    }
};

#endif //REVSURFACE_HPP