#ifndef TEXTURE_H
#define TEXTURE_H

#include <iostream>
#include <vecmath.h>
#include "utils.hpp"
#include "image.hpp"

class Texture {

    Image *image;
    int w, h;
    bool is_texture = false;

public: 
    Texture(const char *filename) {
        image = Image::LoadBMP(filename);
        w = image->Width(), h = image->Height();
        is_texture = true;
    }

    Vector3f getPixel(const Vector2f &pos) {
        double x = pos.x(), y = pos.y();
        x -= floor(x); y -= floor(y);
        x = x > 0. ? x : .99 + x;
        y = y > 0. ? y : .99 + y;
        return image->GetPixel(x * w, y * h);
    }

    double getGray(double u, double v) {
        int pu = (int)(u * w + w) % w, pv = (int)(v * h + h) % h;
        if (pu < 0 || pv < 0)
            fprintf(stderr, "%d %d\n", pu, pv);
        return (image->GetPixel(pu, pv).z() - .5) * 2; 
    }

    double getDisturb(double u, double v, Vector2f &grad) {
        if (!is_texture) { return 0; }
        double disturb = getGray(u, v);
        double du = 1. / w, dv = 1. / h;
        grad[0] = w * (getGray(u + du, v) - getGray(u - du, v)) / 2.0;
        grad[1] = w * (getGray(u, v + dv) - getGray(u, v - dv)) / 2.0;
        return disturb;
    }
};



#endif