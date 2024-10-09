#pragma once

#include "hittable.h"

class Material {
public:
	__device__ Material() { }
	__device__ virtual bool scatter(const Ray& ray, const HitRecord& record, glm::vec3& attenuation, Ray& scattered, curandState* local_rand_state) const {
		return false;
	}
};

class Lambertian : public Material {
public: 
	__device__ Lambertian(glm::vec3& color) : albedo(color) {}

	__device__ virtual bool scatter(const Ray& ray, const HitRecord& record, vec3& attenuation, Ray& scattered, curandState* local_rand_state) const {
		glm::vec3 scatter_direction = record.normal + random_unit_vector(local_rand_state);
		if (glm::length(scatter_direction) <= 1e-8f) {
			scatter_direction = record.normal;
		}
		scattered = Ray(record.point, scatter_direction);
		attenuation = albedo;

		return true;
	}

private:
	glm::vec3 albedo;
};

class Metal : public Material {
public:
	__device__ Metal(const glm::vec3& color, float fuzz) : albedo(color), fuzz(fuzz) {}

	__device__ virtual bool scatter(const Ray& ray, const HitRecord& record, glm::vec3& attenuation, Ray& scattered, curandState* local_rand_state) const{
		glm::vec3 reflected = reflect(ray.direction(), record.normal);
		reflected = glm::normalize(reflected) + (fuzz * random_unit_vector(local_rand_state));
		scattered = Ray(record.point, reflected);
		attenuation = albedo;

		return true;
	}
private:
	glm::vec3 albedo;
	float fuzz;
};

class Dielectric : public Material {
public:
	__device__ Dielectric(float refraction_index) : refraction_index(refraction_index) {}
	
	__device__ virtual bool scatter(const Ray& ray, const HitRecord& record, glm::vec3& attenuation, Ray& scattered, curandState* local_rand_state) const {
		attenuation = vec3(1.0f, 1.0f, 1.0f);

		float ri = record.front_face ? 1.0f / refraction_index : refraction_index;

		glm::vec3 unit_direction = glm::normalize(ray.direction());

		float cos_theta = std::min(glm::dot(-unit_direction, record.normal), 1.0f);
		float sin_theta = std::sqrtf(1.0f - cos_theta * cos_theta);

		bool is_reflect = ri * sin_theta > 1.0f;
		glm::vec3 direction{};

		if (is_reflect || reflectance(cos_theta, ri) > random_float(local_rand_state, 0.0f, 1.0f)) {
			direction = reflect(unit_direction, record.normal);
		}
		else {
			direction = refract(unit_direction, record.normal, ri);
		}

		scattered = Ray(record.point, direction);

		return true;
	}

private:
	float refraction_index;

	__device__ static float reflectance(float cosine, float refraction_index) {
		// Use Schlick's approximation for reflectance.
		float r0 = (1.0f - refraction_index) / (1.0f + refraction_index);
		r0 = r0 * r0;
		return r0 + (1.0f - r0) * std::pow((1.0f - cosine), 5.0f);
	}
};
