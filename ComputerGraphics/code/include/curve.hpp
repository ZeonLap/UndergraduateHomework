#ifndef CURVE_HPP
#define CURVE_HPP

#include "object3d.hpp"
#include <vecmath.h>
#include <vector>
#include <utility>
#include <iostream>

#include <algorithm>

// TODO (PA2): Implement Bernstein class to compute spline basis function.
//       You may refer to the python-script for implementation.

// The CurvePoint object stores information about a point on a curve
// after it has been tesselated: the vertex (V) and the tangent (T)
// It is the responsiblility of functions that create these objects to fill in all the data.

#define T(i) (float)((float)(i) / (float)(n + k + 1))

struct CurvePoint {
    Vector3f V; // Vertex
    Vector3f T; // Tangent  (unit)
};

class Curve : public Object3D {
protected:
    std::vector<Vector3f> controls;
public:
    explicit Curve(std::vector<Vector3f> points) : controls(std::move(points)) {}

    bool intersect(const Ray &r, Hit &h, float tmin) override {
        // TODO
        return false;
    }

    std::vector<Vector3f> &getControls() {
        return controls;
    }

    virtual Vector3f Point(float t) = 0;

    virtual Vector3f Tangent(float t) = 0;

};

class BezierCurve : public Curve {
public:
    explicit BezierCurve(const std::vector<Vector3f> &points) : Curve(points) {
        if (points.size() < 4 || points.size() % 3 != 1) {
            printf("Number of control points of BezierCurve must be 3n+1!\n");
            exit(0);
        }
    }

protected:
    Vector3f Point(float t) override {
        int n = controls.size() - 1;
        std::vector<Vector3f> P(n + 1);
        // Initialize
        for (int i = 0; i <= n; ++i) {
            P[i] = controls[i];
        }
        // 
        for (int k = 1; k <= n; ++k) {
            for (int i = 0; i <= n - k; ++i) {
                P[i] = ((1 - t) * P[i] + t * P[i + 1]);
            }
        }
        return P[0];
    }

    Vector3f Tangent(float t) override {
        int n = controls.size() - 1;
        std::vector<Vector3f> P(n);
        // Initialize
        for (int i = 0; i <= n - 1; ++i) {
            P[i] = n * (controls[i + 1] - controls[i]);
        }
        for (int k = 1; k <= n - 1; ++k) {
            for (int i = 0; i <= n - k - 1; ++i) {
                P[i] = ((1 - t) * P[i] + t * P[i + 1]);
            }
        }
        return P[0];
    }
};

class BsplineCurve : public Curve {
public:
    BsplineCurve(const std::vector<Vector3f> &points) : Curve(points) {
        if (points.size() < 4) {
            printf("Number of control points of BspineCurve must be more than 4!\n");
            exit(0);
        }
    }

protected:
    Vector3f Point(float t) override {
        int n = controls.size() - 1, k = 3;
        int j = floor(t * (n + k + 1));
        std::vector<Vector3f> P(k + 1);
        for (int i = 0; i <= k; ++i) {
            P[i] = controls[j - k + i];
        }

        for (int p = 1; p <= k; ++p) {
            for (int ii = 0; ii <= k - p; ++ii) {
                int i = j - k + p + ii;
                float w1 = (float)(t - T(i)) / (T(i + k - p + 1) - T(i));
                float w2 = 1 - w1;
                P[ii] = w1 * P[ii + 1] + w2 * P[ii];
            }
        }
        return P[0];
    }

    Vector3f Tangent(float t) override {
        int n = controls.size() - 1, k = 3;
        int j = floor(t * (n + k + 1));
        std::vector<Vector3f> D(k);
        for (int l = 0; l < k; ++l) {
            int i = j - k + l;
            D[l] = (float)k / (float)(T(i + k + 1) - T(i + 1)) * (controls[i + 1] - controls[i]);
        }
        for (int l = 1; l < k; ++l) {
            for (int ii = 0; ii < k - l; ++ii) {
                int i = j - k + ii;
                float w1 = (t - T(i + l + 1)) / (T(i + k + 1) - T(i + l + 1));
                float w2 = 1 - w1;
                D[ii] = w1 * D[ii + 1] + w2 * D[ii];
            }
        }
        return D[0];
    }
};

#endif // CURVE_HPP

