#ifndef GROUP_H
#define GROUP_H


#include "object3d.hpp"
#include "ray.hpp"
#include "hit.hpp"
#include "kdtree.hpp"
#include <iostream>
#include <vector>


// TODO: Implement Group - add data structure to store a list of Object*
class Group : public Object3D {

public:

    Group() {
        // ?
    }

    explicit Group (int num_objects) {
        objects.resize(num_objects);
    }

    ~Group() override {
        // ?
    }

    bool intersect(const Ray &r, Hit &h, float tmin) override {
        bool result = false;
        for (int i = 0; i < getGroupSize(); ++i) {
            result |= objects[i]->intersect(r, h, tmin);
        }
        return result;

        // bool result = tree->intersect(r, h, tmin);
        // return result;
    }

    void addObject(int index, Object3D *obj) {
        assert(index < objects.size());
        objects[index] = obj;
        if (index == objects.size() - 1) {
            tree = new KDTree(objects);
        }
    }

    int getGroupSize() {
        return (int)objects.size();
    }

    Object3D* getItem(int index) {
        return objects[index];
    }

private:
    std::vector<Object3D*> objects;
    KDTree *tree;
};

#endif
	
