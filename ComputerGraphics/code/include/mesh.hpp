#ifndef MESH_H
#define MESH_H

#include <vector>
#include "object3d.hpp"
#include "triangle.hpp"
#include "Vector2f.h"
#include "Vector3f.h"
#include "kdtree.hpp"
#include "tiny_obj_loader.h"


class Mesh : public Object3D {

public:
    Mesh(const char *filename, Material *m);

    bool intersect(const Ray &r, Hit &h, float tmin) override;

private:

    KDTree *tree;
    std::vector<Object3D *> triangles;
};

#endif
