# Voxel-Engine
Voxel Engine: High-Performance Voxel-Based 3D Engine in C++

This voxel engine, written in C++ with Vulkan, is designed for high performance and scalability, leveraging low-level control over memory, multi-threading, and GPU-accelerated rendering. It supports seamless voxel-based world generation, where chunks and voxels can be placed at arbitrary world coordinates, allowing for infinite worlds.

## This is NOT a video game
Despite what it looks like, this is not a video game. This is a voxel engine. Most modern day games are built *on top* of engines, leaving the engine to handle low-level graphics, multi-threading, physics stepping, and much more. The focus of my engine is not on gameplay, but to implement these systems myself with a focus on handling and displaying large, block based worlds efficiently. This type of engine can be used in areas like simulations, 3D data visualization, or procedural content creation.

## What is a Voxel Engine; Why is this difficult?
A voxel engine creates and manages 3D worlds made up of volumetric pixels or voxels. This sounds simple, but the primary challenge lies in handling large amounts of data efficiently, especially when dealing with large or complex scenes. Unlike traditional 3D models, which use surfaces and polygons, a voxel engine stores information for every block in a grid, which can quickly become thousands or millions of data entries. The primary goals of my engine are to tackle this problem, while allowing voxels to be dyanamic, be unbounded in position, and performant in a realtime application. Again, this project was developed from near-scratch in c++, using only minimal libraries to accomplish this goal.

## Features
- Infinite, dynamic voxel world generation
- GPU-accelerated rendering using Vulkan
- Multi-threaded architexture: decoupled rendering and update threads
- Custom matrix-based camera controls (supporting mouse and keyboard input)
- Efficient memory management, chunk loading, and unloading
- Optimized for performance, leveraging low-level memory control
- Advanced graphics techniques, including MSAA and mipmapping

## Screenshots and Visuals
![FirstScreenshot](/screenshots/LoweredAnisotropicFiltering.png)

## Technical Highlights
### Low-Level Memory Management
- **Custom memory allocation:** Leveraged C++'s low-level memory control to optimize memory usage. Large voxel data is stored on the heap and passed as references or pointers to avoid blowing up the stack.
- **Chunk streaming:** Implemented efficient chunk loading and unloading, minimizing memory overhead while maintaining smooth gameplay performance in realtime. Chunks are moved onto device local memory for higher performance.

### Multi-Threaded Architecture
- **Decoupled update and rendering threads:** Ths engine uses separate threads for updating world logic and rendering, ensuring consistent frame rates and responsiveness even under heavy load.
- **Thread synchronization:** Implemented thread-safe data handling using mutexes and condition variables to ensure efficient and correct resource access across threads without performance degradation.

### GPU-Accelerated Rendering (Vulkan)
- **Vulkan integration:** Full integration with Vulkan, which gives explicit control over the GPU pipeline for rendering, memory management, and synchronization.
- **MSAA (Multisample Anti-Aliasing):** Integrated MSAA for reducing jagged edges on voxel surfaces. This improves the overall visual quality without significantly affecting rendering performant, thanks to Vulkan's efficient multi-sampling capabilities.
- **Mipmapping for textures:** Implemented mipmapping to optimize the display of textures on distant voxels. This not only improves the visual quality by reducing aliasing in textures but also reduces memory bandwidth usage, as smaller texture versions are used for far-off chunks.
- **Custom shaders:** Created custom GLSL shaders, which leverage push constants, universal buffer objects, and texture atlasing to properly render chunks.

### Custom Camera Controls
- **Input handling:** Captured and processed mouse and keyboard inputs directly, allowing for smooth and intuitive camera controls without relying on extenal libraries.
- **Matrix transformations:** Implemented first-person and free-fly camera controls using matrix transformations calculated from position, yaw, and pitch.  

## Possible Future Plans
- **Lighting:** Lighting could be handled with a rasterization-based deferred rendering system.
- **Greedy Meshing:** This is an algorithm that reduces triangles by combining faces. However, this method disrupts texture atlasing, so I'd need to add vertex attributes, use texture arrays, or use single colored faces.
- **Release as an API:** I could write an API to let users hook into the engine and add their own functionality. It is already partially designed as an engine, with customizations living in my config.h file, but I'd need to write documentation and expose some extra functions.

## Acknowledgements
- My current test textures are a modified version of the **Good Vibes** texture pack by **Acaitart**