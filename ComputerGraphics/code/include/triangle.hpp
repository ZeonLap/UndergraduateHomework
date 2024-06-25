#ifndef TRIANGLE_H
#define TRIANGLE_H

#include "object3d.hpp"
#include <vecmath.h>
#include <cmath>
#include <iostream>
using namespace std;

// TODO: implement this class and add more fields as necessary,
class Triangle: public Object3D {

public:
	Triangle() = delete;

	Triangle( const Vector3f& a, const Vector3f& b, const Vector3f& c, Material* m,
			  const Vector3f& norm_a, const Vector3f& norm_b, const Vector3f& norm_c,
			  const Vector2f& text_a = Vector2f::ZERO, const Vector2f& text_b = Vector2f::ZERO, const Vector2f& text_c = Vector2f::ZERO) : Object3D(m) {
		isInter = true;
		vertices[0] = a;
		vertices[1] = b;
		vertices[2] = c;
		// 
		normal = Vector3f::cross((b - a), (c - a)).normalized();
		double min_arr[3], max_arr[3];
    	for (int i = 0; i < 3; ++i) {
        	min_arr[i] = min(min(a[i], b[i]), c[i]);
        	max_arr[i] = max(max(a[i], b[i]), c[i]);
    	}
    	Vector3f l(min_arr[0], min_arr[1], min_arr[2]), r(max_arr[0], max_arr[1], max_arr[2]);
    	setAABB(l, r);

		// texture settings
		text_pos[0] = text_a;
		text_pos[1] = text_b;
		text_pos[2] = text_c;

		// norm settings
		normals[0] = norm_a;
		normals[1] = norm_b;
		normals[2] = norm_c;

	}

    // a b c are three vertex positions of the triangle
	// 自动计算法向量的版本
	Triangle( const Vector3f& a, const Vector3f& b, const Vector3f& c, Material* m,
			  const Vector2f& text_a = Vector2f::ZERO, const Vector2f& text_b = Vector2f::ZERO, const Vector2f& text_c = Vector2f::ZERO) : Object3D(m) {
		isInter = false;
		vertices[0] = a;
		vertices[1] = b;
		vertices[2] = c;
		// 
		normal = Vector3f::cross((b - a), (c - a)).normalized();
		double min_arr[3], max_arr[3];
    	for (int i = 0; i < 3; ++i) {
        	min_arr[i] = min(min(a[i], b[i]), c[i]);
        	max_arr[i] = max(max(a[i], b[i]), c[i]);
    	}
    	Vector3f l(min_arr[0], min_arr[1], min_arr[2]), r(max_arr[0], max_arr[1], max_arr[2]);
    	setAABB(l, r);

		// texture settings
		text_pos[0] = text_a;
		text_pos[1] = text_b;
		text_pos[2] = text_c;
	}

	bool intersect( const Ray& ray,  Hit& hit , float tmin) override {
		// Implement Crammer's Rule to calculate intersection
		// TODO: Make it faster
        Vector3f E1 = vertices[0] - vertices[1];
		Vector3f E2 = vertices[0] - vertices[2];
		Vector3f S = vertices[0] - ray.getOrigin();
		Vector3f Rd = ray.getDirection();

		Matrix3f M0 = Matrix3f(Rd, E1, E2);
		Matrix3f M1 = Matrix3f(S, E1, E2);
		Matrix3f M2 = Matrix3f(Rd, S, E2);
		Matrix3f M3 = Matrix3f(Rd, E1, S);

		// Cramer
		if (M0.determinant() == 0) {
			return false;
		}
		float t = M1.determinant() / M0.determinant();
		float beta = M2.determinant() / M0.determinant();
		float gamma = M3.determinant() / M0.determinant();

		if (0 < t && 0 < beta && 0 < gamma && beta < 1 && gamma < 1 && beta + gamma < 1 && 
			t < hit.getT() && tmin < t) {
			Vector3f N = (Vector3f::dot(normal, ray.getDirection()) < 0) ? normal : - normal;
			Vector3f _N;
			if (isInter) { _N = get_Norm(ray.pointAtParameter(t)); }
			else { _N = N; }
			hit.set(t, material, _N);
			return true;
		}
		else {
			return false;
		}
	}
	Vector3f normal;
	Vector3f vertices[3];
	Vector2f text_pos[3];
	// 法向量插值
	Vector3f normals[3];
	bool isInter;

	Vector3f get_Norm(const Vector3f &p) {
		Vector3f a = vertices[0] - p, b = vertices[1] - p, c = vertices[2] - p;
		double sa = Vector3f::cross(b, c).length();
		double sb = Vector3f::cross(c, a).length();
		double sc = Vector3f::cross(a, b).length();
		return sa * normals[0] + sb * normals[1] + sc * normals[2];
	}

protected:

};

#endif //TRIANGLE_H
