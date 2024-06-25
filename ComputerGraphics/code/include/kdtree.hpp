#ifndef KDTREE_H
#define KDTREE_H

#include "object3d.hpp"
#include "utils.hpp"
#include <vector>
#include <algorithm>

class Node : public Object3D{

Object3D *lc, *rc;

public:

    Node(Vector3f ld, Vector3f ru, Object3D *_lc, Object3D *_rc) : Object3D(nullptr, ld, ru), lc(_lc), rc(_rc) {

    }

    bool intersect(const Ray &r, Hit &h, float tmin) override {
        if (!aabb.intersect(r)) { return false; }
        else {
            bool res = false;
            res |= lc->intersect(r, h, tmin);
            res |= rc->intersect(r, h, tmin);
            return res;
        }
    }


};

class KDTree {

Object3D *root;

public:

    KDTree(std::vector<Object3D *> objects) {
        root = build(0, 0, objects.size() - 1, objects);
    }

    Object3D *build(int depth, int l, int r, std::vector<Object3D *> &objects) {
        if (l == r) { return objects[l]; }
        int m = (l + r) >> 1, depth_mod_3 = depth % 3;
        // STL: nth_element 
        // Reference : https://www.cnblogs.com/zzzlight/p/14298223.html
        nth_element(
            objects.begin() + l, objects.begin() + m, objects.begin() + r, 
            [depth_mod_3] (Object3D *x, Object3D *y) {
            return x->getAABB().getld()[depth_mod_3] < y->getAABB().getld()[depth_mod_3];
        });

        Object3D *lc = build(depth + 1, l, m, objects), *rc = build(depth + 1, m + 1, r, objects);
        return new Node(minVec(lc->getAABB().getld(), rc->getAABB().getld()), 
                        maxVec(lc->getAABB().getru(), rc->getAABB().getru()),
                        lc, rc);
    }

    bool intersect(const Ray &r, Hit &h, float tmin) {
        return root->intersect(r, h, tmin);
    }
};

#endif