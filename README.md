# Voxel-Engine
Voxel Engine: High-Performance Voxel-Based 3D Engine in C++

This Voxel Engine is developed from near-scratch using Vulkan and GLFW. This engine is designed to efficiently manage and render large-scale voxel worlds, featuring dynamic chunk handling, optimized memory usage, and multi-threaded performance. With a focus on flexibility and scaling, the system is capable of handling vast, procedurally generated terrains or complex voxel environments.

## This is NOT a video game
Despite what it looks like, this is not a video game. This is a voxel engine. Most modern day games are built *on top* of engines, leaving the engine to handle low-level graphics, multi-threading, physics stepping, and much more. The focus of my engine is not on gameplay, but to implement these systems myself with a focus on handling and displaying large, block based worlds efficiently. This engine can be used in areas like simulations, 3D data visualization, or procedural content creation.

## What is a Voxel Engine; Why is this difficult?
A voxel engine creates and manages 3D worlds made up of tiny blocks(voxels). This sounds simple, but the challenge lies in handling **huge amounts of data** efficiently, especially when dealing with large or complex scenes. Unlike traditional 3D models, which use surfaces and polygons, a voxel engine stores information for every block in a grid, which can quickly become **millions of points.**

## Preview: Screenshots and Visuals

## Preview Builds

## Features
### Built with **Vulkan** & **GLFW**
The engine is constructed using **Vulkan**, a low-level graphics API, and **GLFW** for window and input management. Vulkan provides fine control over the rendering pipeline and resource management, allowing the engine to maximize GPU performance.
- **Explicit Resource Management:** Vulkan’s API allows precise control over memory and buffer allocation, synchronization primitives, and resource binding.
- **Low-Level Pipeline Control:** The rendering process is optimized for performance through manual configuration of pipeline states, shaders, and frame synchronization.

More to be added soon! I've implemented much more such as multi-threading and fixed physics steps, but need a second to update this readme.

### Technical Highlights
- **Low-Level Vulkan Integration:** Direct control over the Vulkan API enables advanced optimization techniques for rendering and memory management.
- **Memory and Resource Efficiency:** By carefully managing Vulkan memory, buffer allocations, and synchronization, the engine minimizes latency and maximized GPU throughput.
- **Debugging and Profiling Tools:** Extensive use of Vulkan's validation layers and performance profiling tools ensures that the engine runs efficiently and smoothly.

### Future Plans
- **Lighting:** Lighting will be handled with a rasterization-based deferred rendering system.
- **Greedy Meshing:** This is an optimization algorithm to combine faces to reduce this number of triangles.
- **Release as an API:** I'd like to write an API to let users hook into the engine and add their own functionality; this covers up the engine backend reducing complexity for the average user.

## Acknowledgements
- my current test textures are a modified version of the **Good Vibes** texture pack by **Acaitart**