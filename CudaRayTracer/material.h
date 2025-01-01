#pragma once

#include "hittable.h"
#include "texture.h"

class Material {
public:
	__device__ __host__ Material() { }

	__device__ virtual vec3 emitted(float u, float v, const vec3& p) const {
		return vec3(0, 0, 0);
	}

	__device__ virtual bool scatter(const Ray& ray, const HitRecord& record, vec3& attenuation, Ray& scattered, curandState* local_rand_state) const {
		return false;
	}
};

class Lambertian : public Material {
public: 
	//__device__ __host__ Lambertian(const vec3& color) : albedo(color) {}
	__device__ __host__ Lambertian(const vec3& color) : texture(new SolidColor(color)) {}
	__device__ __host__ Lambertian(Texture* texure) : texture(texture) {}
	__device__ virtual bool scatter(const Ray& ray, const HitRecord& record, vec3& attenuation, Ray& scattered, curandState* local_rand_state) const {
		vec3 scatter_direction = record.normal + random_unit_vector(local_rand_state);
		if (glm::length(scatter_direction) <= 1e-8f) {
			scatter_direction = record.normal;
		}
		scattered = Ray(record.point, scatter_direction);
		attenuation = texture->value(record.u, record.v, record.point);

		return true;
	}

private:
	Texture* texture;
};

class Metal : public Material {
public:
	__device__ __host__ Metal(const vec3& color, float fuzz) : albedo(color), fuzz(fuzz) {}

	__device__ virtual bool scatter(const Ray& ray, const HitRecord& record, vec3& attenuation, Ray& scattered, curandState* local_rand_state) const{
		vec3 reflected = reflect(ray.direction(), record.normal);
		reflected = glm::normalize(reflected) + (fuzz * random_unit_vector(local_rand_state));
		scattered = Ray(record.point, reflected);
		attenuation = albedo;

		return true;
	}
private:
	vec3 albedo;
	float fuzz;
};

class Dielectric : public Material {
public:
	__device__ __host__ Dielectric(float refraction_index) : refraction_index(refraction_index) {}
	
	__device__ virtual bool scatter(const Ray& ray, const HitRecord& record, vec3& attenuation, Ray& scattered, curandState* local_rand_state) const {
		attenuation = vec3(1.0f, 1.0f, 1.0f); // No attenuation for dielectric materials

		// Determine the refractive index depending on the front/back face of the hit surface
		float ri = record.front_face ? 1.0f / refraction_index : refraction_index;

		// Normalize the incoming ray direction
		vec3 unit_direction = glm::normalize(ray.direction());

		// Calculate the cosine of the angle between the incident ray and the normal
		float cos_theta = glm::min(glm::dot(-unit_direction, record.normal), 1.0f);
		float sin_theta = glm::sqrt(1.0f - cos_theta * cos_theta);

		// Check for total internal reflection (TIR) condition
		bool is_reflect = ri * sin_theta > 1.0f;

		vec3 direction;
		if (is_reflect || reflectance(cos_theta, ri) > random_float(local_rand_state, 0.0f, 1.0f)) {
			// Total internal reflection or probabilistic reflection based on Schlick's approximation
			direction = reflect(unit_direction, record.normal);
		}
		else {
			// Refract the ray based on Snell's Law (using refractive indices)
			direction = refract(unit_direction, record.normal, ri);
		}

		// Slight offset from the intersection point to avoid self-intersection
		scattered = Ray(record.point + direction * 1e-3f, direction); // Slightly offset to avoid self-intersection

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

class DiffuseLight : public Material{
public:
	__device__ __host__ DiffuseLight(Texture* tex) : texture(tex) {}
	__device__ __host__ DiffuseLight(const vec3& emit) : texture(new SolidColor(emit)) {}

	__device__ vec3 emitted(float u, float v, const vec3& p) const override {
		return texture->value(u, v, p);
	}
private:
	Texture* texture;
};
