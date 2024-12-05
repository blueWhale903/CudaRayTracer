# SIMPLE RAY TRACING (C/C++ & CUDA)

This project is a custom ray tracing engine developed using C/C++ and CUDA, designed to explore the fundamentals of ray tracing techniques for high-quality image rendering. The engine is optimized for GPU acceleration using CUDA, providing significant performance improvements over CPU-based implementations.

![500spp-light](https://github.com/user-attachments/assets/4e37a612-b166-43f6-ae89-b8c600b0de2c)

# Features

- CUDA Acceleration: The ray tracing calculations are offloaded to the GPU, speeding up the rendering process.
- Basic Ray Tracing Pipeline: Implements a basic ray tracing pipeline with ray-object intersection, lighting, and shading.
- Bounding Volume Hierarchy (BVH): Efficient acceleration structure used for fast intersection tests between rays and objects.
- Materials: Supports multiple types of materials such as Lambertian (diffuse), Metal, and Emissive lights.
- Simple Scene Setup: Allows for easy creation of scenes with basic geometric shapes (triangles) and textures.

# Requirements

- CUDA Toolkit: Ensure you have the CUDA Toolkit installed and configured on your system.
- C++ Compiler: A C++ compiler (e.g., GCC, Clang, or MSVC).
- CMake: For building and managing the project.
- Assimp (optional): For model loading from .obj or other 3D file formats.

# Some results

![sample](https://github.com/user-attachments/assets/d26daddb-9b17-4958-888e-7c9c7a132e42)

![dragon](https://github.com/user-attachments/assets/e7601c7b-927f-4a52-8349-28d179a7ada3)

