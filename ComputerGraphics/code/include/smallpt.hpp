#ifndef SMALLPT_H
#define SMALLPT_H


#include <vecmath.h>
#include <cmath>
#include <iostream>
#include </opt/homebrew/Cellar/libomp/16.0.6/include/omp.h>
#include "object3d.hpp"
#include "group.hpp"
#include "camera.hpp"
#include "scene_parser.hpp"
#include "image.hpp"
#include "utils.hpp"

inline double max(const Vector3f &v) {
    return (
        (v.x() > v.y() && v.x() > v.z()) ? v.x() :
        ((v.y() > v.z()) ? v.y() : v.z())
    );
}

inline double clamp(double x) { return x < 0 ? 0 : x > 1 ? 1 : x; }
inline double toDouble(double x) { return (double)(pow(clamp(x), 1/2.2) * 255 + .5) / 255.; }

class SmallPT {

public:
    // 参见"Global Illumination in 99 lines of C++"中的radiance函数
    Vector3f Radiance(const Ray &r, int depth, Group *group) {
        Hit hit;
        bool isIntersect = group->intersect(r, hit, 1e-3);
        /* 不相交，直接返回环境光 */
        if (!isIntersect) { return Vector3f(0, 0, 0); }

        Material *material = hit.getMaterial();
        Vector3f x = r.pointAtParameter(hit.getT());
        Vector3f nl = hit.getNormal();
        Vector3f n = hit.getInto() ? nl : -nl;
        Vector3f f = hit.getMaterial()->_getDiffuseColor(hit);
        double p = max(f);

        if (++depth > 5) {
            if (rand01() < p) { f *= 1 / p; }
            else { return material->getAmbientColor(); }
        }
        /* 漫反射 */
        if (material->getReflectionType() == DIFF) {
            double r1 = 2 * M_PI * rand01(), r2 = rand01(), r2s = sqrt(r2);
            Vector3f w = nl;
            Vector3f u = Vector3f::cross((fabs(w.x()) > .1 ? Vector3f::UP : Vector3f::RIGHT), w).normalized();
            Vector3f v = Vector3f::cross(w, u); /* 一定是单位向量 */
            Vector3f d = (u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrt(1 - r2)).normalized();
            return material->getAmbientColor() + f * Radiance(Ray(x, d), depth, group);
        }
        /* 镜面反射 */
        else if (material->getReflectionType() == SPEC) {
            return material->getAmbientColor() + f * Radiance(Ray(x, r.getDirection() - n * 2 * Vector3f::dot(n, r.getDirection())), depth, group);
        }
        /* 折射 */
        else {
            Ray reflRay(x, r.getDirection() - 2 * Vector3f::dot(n, r.getDirection()) * n);
            bool into = hit.getInto();
            double nc = 1, nt = material->getRefractiveIndex();
            double nnt = into ? nc / nt : nt / nc, ddn = Vector3f::dot(r.getDirection(), nl);
            double cos2t = 1 - nnt * nnt * (1 - ddn * ddn);
            /* 全反射 */
            if (cos2t < 0) { return material->getAmbientColor() + f * Radiance(reflRay, depth, group); }
            Vector3f tdir = (r.getDirection() * nnt - n * (into ? 1 : -1) * (ddn * nnt + sqrt(cos2t))).normalized();
            double a = nt - nc, b = nt + nc, R0 = a * a / (b * b), c = 1 - (into ? -ddn : Vector3f::dot(tdir, n));
            double Re = R0 + (1 - R0) * c * c * c * c * c, Tr = 1 - Re, P = .25 + .5 * Re, RP = Re / P, TP = Tr / (1 - P);
            return (
                material->getAmbientColor() + f * (depth > 2 ? (rand01() < P ?
                Radiance(reflRay, depth, group) * RP : Radiance(Ray(x, tdir), depth, group) * TP) :
                Radiance(reflRay, depth, group) * Re + Radiance(Ray(x, tdir), depth, group) * Tr)
            );
        }
    }

    Image PathTrace(SceneParser &sceneParser, int samples) {
        Camera *cam = sceneParser.getCamera();
        int w = cam->getWidth(), h = cam->getHeight();
        Image image = Image(w, h);
        Group *group = sceneParser.getGroup();
        auto start = omp_get_wtime();
        int threads = omp_get_num_threads();
        int finished_pixels = 0;
        #pragma omp parallel for schedule(dynamic, 1)
        for (int y = 0; y < h; ++y) {
            fprintf(stderr, "\rRendering (%d spp) %5.2f%%, Time %.2fs", samples * 4, 100. * finished_pixels / (h * w), omp_get_wtime() - start);
            for (int x = 0; x < w; ++x) {
                Vector3f pixel;
                /* 子像素采样 */
                for (int sy = 0; sy < 2; ++sy) {
                    for (int sx = 0; sx < 2; ++sx) {
                        Vector3f r;
                        for (int s = 0; s < samples; ++s) {
                            double r1 = 2 * rand01(), dx = r1 < 1 ? sqrt(r1) - 1 : 1 - sqrt(2 - r1);
                            double r2 = 2 * rand01(), dy = r2 < 1 ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);
                            double cur_x = x + sx - 0.5 + dx, cur_y = y + sy - 0.5 + dy;
                            r = r + Radiance(cam->generateRay(Vector2f(cur_x, cur_y)), 0, group) * (1. / samples);
                        }
                        pixel += Vector3f(toDouble(clamp(r.x())), toDouble(clamp(r.y())), toDouble(clamp(r.z()))) * .25;
                    }
                }
                image.SetPixel(x, y, pixel);
            }
            finished_pixels += w;
        }
        return image;
    }
};

#endif