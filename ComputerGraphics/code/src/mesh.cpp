#define TINYOBJLOADER_IMPLEMENTATION

#include "mesh.hpp"
#include "triangle.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <utility>
#include <sstream>

bool Mesh::intersect(const Ray &r, Hit &h, float tmin) {
    return tree->intersect(r, h, tmin);
}

Mesh::Mesh(const char *filename, Material *material) : Object3D(material) {
    tinyobj::ObjReader reader;

    if (!reader.ParseFromFile(filename)) {
        if (!reader.Error().empty()) {
            std::cerr << "TinyObjReader: " << reader.Error();
        }
        exit(1);
    }

    if (!reader.Warning().empty()) {
        std::cout << "TinyObjReader: " << reader.Warning();
    }

    auto shapes = reader.GetShapes();
    auto attrib = reader.GetAttrib();

    // Loop over shapes
    for (size_t s = 0; s < shapes.size(); ++s) {
        size_t index_offset = 0;
        // Loop over faces
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); ++f) {
            size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);
            // assert(fv == 3);
            Vector3f vertices[3], normal[3];
            Vector2f texcoord[3];
            // Loop over vertices in the face.
            bool hasN = false;
            for (size_t v = 0; v < 3; ++v) {
                // access to vertex
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
                tinyobj::real_t vx = attrib.vertices[3 * size_t(idx.vertex_index) + 0];
                tinyobj::real_t vy = attrib.vertices[3 * size_t(idx.vertex_index) + 1];
                tinyobj::real_t vz = attrib.vertices[3 * size_t(idx.vertex_index) + 2];
                vertices[v] = Vector3f(vx, vy, vz);

                // Check if `normal_index` is zero or positive. negative = no normal data
                if (idx.normal_index >= 0) {
                    hasN = true;
                    tinyobj::real_t nx = attrib.normals[3 * size_t(idx.normal_index) + 0];
                    tinyobj::real_t ny = attrib.normals[3 * size_t(idx.normal_index) + 1];
                    tinyobj::real_t nz = attrib.normals[3 * size_t(idx.normal_index) + 2];
                    normal[v] = Vector3f(nx, ny, nz);
                }

                // Check if `texcoord_index` is zero or positive. negative = no texcoord data
                if (idx.texcoord_index >= 0) {
                    tinyobj::real_t tx = attrib.texcoords[2 * size_t(idx.texcoord_index) + 0];
                    tinyobj::real_t ty = attrib.texcoords[2 * size_t(idx.texcoord_index) + 1];
                    texcoord[v] = Vector2f(tx, ty);
                }
            }
            // 
            if (hasN) {
                this->triangles.push_back(new Triangle(vertices[0], vertices[1], vertices[2], this->material, normal[0], normal[1], normal[2], texcoord[0], texcoord[1], texcoord[2]));
            } else {
                this->triangles.push_back(new Triangle(vertices[0], vertices[1], vertices[2], this->material, texcoord[0], texcoord[1], texcoord[2]));
            }
            
            index_offset += fv;

            // per-face material
            shapes[s].mesh.material_ids[f];
        }
    }

    // Build Tree
    tree = new KDTree(this->triangles);
}
