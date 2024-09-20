# Voxel-Engine
Voxel Engine: High-Performance Voxel-Based 3D Engine in C++

This Voxel Engine is developed from near-scratch using **Vulkan** and **GLFW**. This engine is designed to efficiently manage and render large-scale voxel worlds, featuring dynamic chunk handling, optimized memory usage, and multi-threaded performance. With a focus on scalability and flexibility, the system is capabale of handling vast, procedurally generated terrains or complex voxel environments.

This project is still in progress, but most systems and algorithms have been implemented already.

## Features
### Built with **Vulkan** & **GLFW**
The engine is constructed using **Vulkan**, a low-level graphics API, and **GLFW** for window and input management. Vulkan provides fine control over the rendering pipeline and resource management, allowing the engine to maximize GPU performance.
- **Explicit Resource Management:** Vulkan’s API allows precise control over memory and buffer allocation, synchronization primitives, and resource binding.
- **Low-Level Pipeline Control:** The rendering process is optimized for performance through manual configuration of pipeline states, shaders, and frame synchronization.

### Efficient Chunk & Voxel System
The core of the engine is an optimized **chunk-based voxel system** that dynamically loads, unloads, and renders multiple chunks in real time.
- **Dynamic Chunk Management:** Chunks are loaded into memory based on the camera's position and viewpoint, with unused chunks efficiently culled to conserve resources.
- **Memory Optimization:** Chunks are organized to make allocations from device memory as few times as possible.

### Technical Highlights
- **Low-Level Vulkan Integration:** Direct control over the Vulkan API enables advanced optimization techniques for rendering and memory management.
- **Memory and Resource Efficiency:** By carefully managing Vulkan memory, buffer allocations, and synchronization, the engine minimizes latency and maximized GPU throughput.
- **Debugging and Profiling Tools:** Extensive use of Vulkan's validation layers and performance profiling tools ensures that the engine runs efficiently and smoothly.

### Future Plans
- **Multi-Threading:** Two threads should be used to maximize performance. Rendering should be on its own thread, and game logic and state updates should be handles on a separate thread.
- **Lighting:** Lighting will be handled with a rasterization-based deferred rendering system.
- **Player Interaction:** A player will be added to allow camera movement and placing or destroying blocks.