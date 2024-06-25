#ifndef MATERIAL_H
#define MATERIAL_H

#include <cassert>
#include <vecmath.h>

#include "ray.hpp"
#include "hit.hpp"
#include "texture.hpp"
#include <iostream>

#define DIFF    0  // 漫反射
#define SPEC    1  // 镜面反射
#define REFR    2  // 折射

class Material {
public:

    explicit Material(const char *filename, /* 纹理 */ const char *bumpname, /* 凹凸 */
                      const Vector3f &a_color, 
                      const Vector3f &d_color, 
                      const Vector3f &s_color = Vector3f::ZERO, 
                      int refl_t = DIFF, double refr_index = 1.0 /* 折射率 */) :
            ambientColor(a_color), diffuseColor(d_color), specularColor(s_color), reflectionType(refl_t), refractiveIndex(refr_index) {
        if (filename[0] == '\0') { texture = nullptr; }
        else {
            texture = new Texture(filename);
        }
        if (bumpname[0] == '\0') { bump = nullptr; }
        else {
            bump = new Texture(bumpname);
        }
    }

    virtual ~Material() = default;

    virtual Vector3f getDiffuseColor() const {
        return diffuseColor;
    }

    Vector3f _getDiffuseColor(const Hit &h) const {
        if (texture != nullptr) { 
            return texture->getPixel(h.getPos()); 
        }
        else { return diffuseColor; }
    }

    virtual Vector3f getAmbientColor() const {
        return ambientColor;
    }

    virtual int getReflectionType() const {
        return reflectionType;
    }

    virtual double getRefractiveIndex() const {
        return refractiveIndex;
    }

    Texture *getTexture() {
        return texture;
    }

    Texture *getBump() {
        return bump;
    }

protected:
    int reflectionType;
    double refractiveIndex;
    Vector3f ambientColor;
    Vector3f diffuseColor;
    Vector3f specularColor;
    Texture *texture, *bump;
};


#endif // MATERIAL_H
