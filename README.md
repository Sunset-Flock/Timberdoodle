# Timberdoodle

Home of colaboration grown graphics engine by Matěj Sakmary and Patrick Ahrens.

This engine is used as a basis for our research projects and general education in graphics programming.

explicit goals of this project:
* comment features and abstractions with blocks of text in the respective files
* make big comments explaining a complex part of the engine in place
* properly consider, propagate and handle most runtime errors
* be consistent
* refactor regularly after gaining greater understanding of an aspect of the engine
* DO NOT overgeneralize

We will try to keep the engine very well maintainable. This will include many partial rewrites.

# Screenshots
![](https://github.com/Sunset-Flock/Timberdoodle/blob/main/media/battle_scene_main.png)
![](https://github.com/Sunset-Flock/Timberdoodle/blob/main/media/night_bistro.png)
![](https://github.com/Sunset-Flock/Timberdoodle/blob/main/media/day_bistro.png)

## Features
### Virtual Shadow Maps (Matěj Sakmary)
![](https://github.com/Sunset-Flock/Timberdoodle/blob/main/media/vsm_overview.png)

A full article describing this technique is scheduled to release in GPU Zen 3. In the meantime there is a great [blogbost](https://ktstephano.github.io/rendering/stratusgfx/svsm) by J Stephano which describes a portion of the implementation. Please note that the version described by him has been extended with in this implementation for essential features such as page caching and others.

Virtual shadow maps are a technique initally adapted by Unreal Engine 5. As the name suggests, VSMs decouple the logical, or \textit{virtual}, shadow map from its physical backing. In our implementation, a VSM consists of 16 overlapping 4096x4096 cascades arranged in a concentric manner. Every cascade consists of a sparsely allocated set of pages, with each page spanning 128x128 texels. Each cascade's frustum is twice the size of the previous cascade's. The size of the first cascade, specified in world-space units, thus determines the resolution and quality of the entire VSM. In our implementation, we chose to make our first cascade 2x2 meters wide, with the last cascade covering a diameter of 65km. The virtual resolution of cascades, their number, and the resolution of pages can vary.

Two main textures/tables are used to keep track of the VSM. A *virtual page table* (VPT) is used to map virtual addresses to physical addresses when we sample the shadow map. The VPT also contains metadata: page allocation status, visibility status, and a dirty flag. The allocation status flag marks pages that are backed by a physical page. The visibility status flag indicates whether the page is visible to the viewer in this frame. The dirty status flag indicates that the physical page contents are no longer valid and should be redrawn. We further define a *physical page table* (PPT) which provides the mapping of physical pages to virtual pages (the inverse of the VPT).

Sparse (or tiled) resources in graphics APIs initially seem like an obvious choice for VSMs. However, the API for sparse resources does not allow for an indirect GPU-driven approach. To know which VSM pages need to be backed by physical memory would require GPU readback. This adds unacceptable latency and leads to shadow pop-in. Our implementation opts for a manual allocation scheme that allows for execution of every part of the algorithm on the GPU and avoids this round trip latency.

Caching is a common and important optimization for shadow mapping in the semi-static scenes frequently found in real-time applications. We implement a per-page caching scheme in VSM. After a page is drawn and sampled, it is not discarded. Instead, its contents are preserved, and the allocation is left untouched, continuing its lifetime into the next frame. By doing this, we minimize the number of required draws and allow for more efficient culling. To support caching, we modify the positioning of the individual cascade frustums so that they no longer are strictly centered at the viewer's position. They still follow the main camera position; however, the change in their position occurs at page-sized offsets. Further, the light position is constrained so that, when modified, it slides along a plane parallel to the near-plane of the respective light matrix. This constraint is necessary for the data stored in cached pages to remain valid even after translating the light matrix.

As the cascade frustum moves to follow the main camera, new pages, previously located on the edge just outside of the cascade frustum, might need to be drawn. In order to preserve previously cached pages, we utilize a concept called *2D wraparound addressing*. As new pages enter the frustum when it moves, old pages exit. With wraparound addressing, the newly entered pages are mapped to the location in the VPT previously occupied by old, exiting pages. This requires us to store a per-cascade offset of the respective light matrix position from the origin. The wrapped coordinates are then used to lookup VPT entries.

The process of drawing VSM consists of two main stages - Bookkeeping and Drawing. During bookkepping the depth buffer is first analyzed and the set of visible tiles is determined. Following this, tiles not visible in the previous frame are allocated from the memory texture backing the VSM. Additionally, previously cached tiles invalidated by dynamic object movement are also marked for the draw pass. After determining which virtual pages are visible to the viewer and allocating physical pages to back them, the scene is rendered to each cascade. Due to the VSM having many cascades, it is critical to have effective and granular scene culling. In a typical frame, only a fraction of the pages in each cascade will be marked as dirty at any time. With this knowledge, we define a structure called the hierarchical page buffer (HPB). Similarly to how a hierarchical Z buffer (HiZ) is built from the depth buffer for the purpose of occlusion culling, we build HPB from each cascade VPT for the purpose of culling geometry that does not overlap any dirty page. Thus, each HPB level is constructed as a 2x2 reduction of the lower level, with the lowest level reducing the VPT itself. Contrary to HiZ, we do not reduce the depth. Instead, we do a max reduction of the dirty status flag marking the page as being drawn into this frame.

### Skybox (Matěj Sakmary)
The skybox is a slightly improved version of the one described in my [bachelors thesis](https://github.com/MatejSakmary/atmosphere-bac). The base technique stems from a great article from Sebastien Hillaire [A Scalable and Production Ready Sky and Atmosphere Rendering Technique](https://sebh.github.io/publications/egsr2020.pdf). The main idea is to precompute a set of two dimensional look up tables which then speed up the raymarch of the atmosphere itself. The atmosphere is rendered into a low dimensional look up table which is then upscaled during shading of the skybox. 

To generate the lighting contribution of the skybox it is each frame convolved into a cubemap describing the integrated light contribution in a specific direction. To avoid noisy cubemap while keeping minimal time complexity, the convolved cubemap is temporalily accumulated using exponential running awerage. This cubemap is then sampled during shading using the surface normal to determine the indirect skybox lighting.

### Asset Loading (Matěj Sakmary & Patrick Ahrens)
The asset loader is fully asychronous and utilizes built in threadpool implementation. We utilize a concept called manifest to keep track of mesh and asset metadata. The manifest is strictly growing set of descriptors filled when an asset is loaded for the first time of the runtime of the application. Only gltf file format is currently supported for the asset loader. Once the manifest is populated a job per texture and per mesh is inserted into the threadpool. Each job is thus responsible for the loading, processing and uploading of the respective resource to the GPU. Once finished the manifest is further updated with the runtime data - the pointers to the buffers and or textures containing the relevant information. This is picked up by the shading and as soon as the data is awailable to the GPU the mesh is drawn and textured.

The threadpool is a very simple two priority queue data structure utilizing condition variables to avoid any busy waiting and wasting of resources. While it is currently only used for asset loading it has been designed with universality in mind. Thus parallel command recording and or shader compilation are planned for the future.

### Two pass Meshlet culling (Patrick Ahrens)
