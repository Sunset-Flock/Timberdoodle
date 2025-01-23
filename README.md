# Timberdoodle

Home of colaboration grown graphics engine by Matěj Sakmary and Patrick Ahrens.

This engine is used as a basis for our research projects and general education in graphics programming.

# Screenshots

![](https://github.com/Sunset-Flock/Timberdoodle/blob/main/media/Showcase0_Bistro.png)
![](https://github.com/Sunset-Flock/Timberdoodle/blob/main/media/Showcase1_Sponza.png)
![](https://github.com/Sunset-Flock/Timberdoodle/blob/main/media/Showcase2_Battle.png)

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

### Debug views (Matěj Sakmary & Patrick Ahrens)
As we believe debug views are very important and helpful when idetifying and fixing bugs, Timberdoodle includes several drawing modes visualizing so far implemented techniques. As we fully adopted meshlet workflow to allow for efficient culling, we include a meshlet, mesh, entity and triangle views. To visualize the effects of culling we include overdraw visualizations for both main pass as well as shadowmaps. For Shadowmaps an additional debug view (shown at the top of the Virtual Shadow Maps section) is also included. This debug view visualizes individual cascades and pages. Another very important feature we implemented is the observer camera. This feature allows to freeze the main camera and detach another, observer, camera which can freely move around the scene and inspect its state as if it has been drawn from the main camera. This option was priceless when debugging various artifacts which were hard to argue about from the point of view of the main camera but were immediatelly made clear when inspected using the observer.

Additionally to debug views Timberdoodle also includes additional set of debug textures used for various purposes. One example is the presence of textures showing the state of the VSM memory pool as well as the state of each of the cascade VPTs. To further allow for easy debugging, the option do draw simple shapes such as rectangles, AABBs, cubes, spheres and other are built into the core of the engine. The debug shape drawing is supported both from the CPU as well as the GPU. This allows us to immediatelly visualize wide spectrum of data and quickly gain deeper insight into potential issues.

![](https://github.com/Sunset-Flock/Timberdoodle/blob/main/media/overdraw_bistro.png)
![](https://github.com/Sunset-Flock/Timberdoodle/blob/main/media/meshlet_bistro.png)
![](https://github.com/Sunset-Flock/Timberdoodle/blob/main/media/culling_bistro.png)

### Visbuffer Geometry Pipeline (Patrick Ahrens)

Tido uses a gpu driven fully bindless geometry pipeline to render a visbuffer (triangle id + depth).

#### Mesh Preparation

When loading assets, tido preprocesses all meshes:
1. optimize vertices of each mesh
2. auto generate lods for each mesh
    - each lod is roughly half the triangle count of the previous lod
    - uses error metric based on normal and positional error for each simplification
    - up to 16 lods are generated, the generation of lods is stopped after crossing an error threshold
3. generate meshlets for each mesh
    - up to 64 triangles and 64 vertices per meshlet
    - compressed indexbuffer for each meshlet

Most of these operations use the meshoptimizer library

Each mesh with all its data (indices, vertices meshlets ...) is packed into a single buffer allocation. A GPUMesh struct is created holding metadata (vertex count, aabb...) and pointers to the section of the mesh buffer containing lists of all the meshes attributes (indices, meshlets, vertices...).

> In tido we exclusively use Buffer Device Address pointers and never buffer objects in shader code

This GPUMesh struct is then put into a buffer (called mesh manifest) containing an array with all GPUMeshes. This buffer is available in all shaders in shading, allowing any rendering shader to always access all geometry data.

#### Beginning of Frame on CPU

At the beginning of each frame, tido goes over all entities in the scene and selects an appropriate lod.
The lod is based on a rough pixel error metric. Each each lods bounding box is projected into a rotated view towards the lod. Rotating the view for the projection ensures that there is never an lod change based on the real cameras orientation, as well as making the selection consistent with raytracing.

The projected bounding boxes pixel size is multiplied with the estimated error calculated by the simplification. The lod with the highest error that still falls under ~1 pixel error is selected.
This scheme is usually good enough to ensure that lod changes are not visible.

Using the selected lods the new frames TLAS (used in raytracing) is build and a list of MeshInstances is generated. This list is then send to the gpu each frame.

#### The Mega Drawcall

Reguardless of mesh instance count, tido can draw all geometry in one drawcall. This means extremely low cpu overhead for any amount of geometry on the screen.

To achieve this, tido must perform multiple phases of "work expansion":
1. a compute shader goes over all mesh instances
    - culls meshes (described later)
    - enqueues the surviving meshes meshlets into a "work expansion" buffer
2. a indirect compute shader goes over all surviving meshlets using the "work expansion" buffer
    - culls meshlets 
    - writes surviving meshlets into a meshlet instance buffer 
3. a indirect mesh shader dispatch goes over all meshlet instances and draws them.
    - uses the prepared meshlets
    - draws visibility id and depth
    - visibility id is 24 bit meshlet instance index and 6 bits triangle index 2 bit to spare
    - uses inverse z to get best precision and infinite far plane
    - as all geometry is bindless, the indirections from meshlet_instance->mesh_instance->mesh allows this to be one drawcall

The second step can also be performed by task shaders instead of a compute pass. Sadly, task shaders have poor performance for very high meshlet counts. This makes them slower on my gpu (RTX4080) for high meshlet counts. For lowish meshlet counts (<100k) the task shaders are faster tho because they avoid the barrier between compute and mesh shader as well as allow for some overlap between meshshading and meshlet culling work. 

There is a catch to the "one drawcall" to draw everything, as the Pixelshader for alpha discard geometry prevents early z tests due to the discard call. To prevent general performance degradation, Tido has two rasterization pipelines, one without alpha discard that can perform early z tests used on opaque. And one pipeline that does use discard to support alpha discard geometry.

References: 
* https://developer.nvidia.com/blog/introduction-turing-mesh-shaders/
* https://gist.github.com/Ipotrick/d6bcf9696b00c72eb037370794733de8
* https://github.com/zeux/meshoptimizer
* http://filmicworlds.com/blog/visibility-buffer-rendering-with-material-graphs/

#### GPU Culling

Now this is one of two phases tido performs to draw. Tido performs two pass occlusion culling.
This means Tido performs the whole draw pipeline twice.

First culling everything against the depth of last frame and then culling everything that was not drawn in the first pass again against the newly generated depth in the first pass. To avoid culling things that were drawn in the first pass again, tido builds a bitfield with indirections from entity->mesh_instance->meshlet_instance to save space (2mb is enough for all tested scenes).

Mesh and meshlet culling works essentially the same: cull against frustum then against occlusion.

As tido uses mesh shading it can also perform shader based triangle culling. Tido culls large triangles against depth and all other triangles when its either a backface or a "microtriangle" (triangle that is so small and in just the right position, its not rasterized).

The hierarchical depth buffer (called HiZ in tido) is generated in a single pass with a custom made downsampler similar to AMDs SPD. 

The depth culling tests from 2x2 up to 4x4 texels in the HiZ depending on size and position of the projected aabb of the test object. 

Tido has an alternative culling mode, where it instead of culling against previous frames depth and then drawing for the first pass, simply directly draws all meshlets that were visible in the last frame.
For scenes with larger triangles this can be up to 25% faster. The downside is that this generally leaves larger "holes" in the HiZ as effectively using the meshlets from last frame instead of depth is a much finer culling. This can lead to much worse performance when moving the camera with high poly scenes.

References: 
* https://github.com/GPUOpen-Effects/FidelityFX-SPD

#### Visbuffer

Just using the visbuffer, we can fully reconstruct all triangle information in a deferred manner when shading later. 

This is fully bindless, any shader reading it can fully reconstruct all triangle information including barycentrics and derivatives.

As mentioned earlier, tidos alternative mode draws the previous frames meshlets first instead of using previous depth to cull and darw. To find all visible meshlets from the previous frame, tido performs a compute dispatch over the visbuffer analyzing it. This shader builds a list of all unique meshlets present in the visbuffer. Its generally very fast (<0.04ms).
This list of visible meshlets could also be used to draw the scene again but with meshlet perfect culling.
This allows for very efficient f+ drawing.

References: 
* http://filmicworlds.com/blog/visibility-buffer-rendering-with-material-graphs/
* https://research.activision.com/publications/2021/09/geometry-rendering-pipeline-architecture

### Probe Based Global Illumination (PGI)

![](https://github.com/Sunset-Flock/Timberdoodle/blob/main/media/shaded.png)
![](https://github.com/Sunset-Flock/Timberdoodle/blob/main/media/probes.png)

Based on [RTXGI-DDGI](https://github.com/NVIDIAGameWorks/RTXGI-DDGI).

The world is covered in an irradiance volume represented as a probe grid. Rays are shot from each probe, consine convolved and written to the texels of the probe. 
Each probe is also calculating a visibility term via a statistical depthmap (average depth and depth variance per texel) to avoid leaking.
In practice this works well and can be used as a unified indirect lighting model as it is volumetric and can be applied to all dynamic, static and volumetric rendering.

There are a few key improvements made to improve the performance and scaling of DDGI:
* Variable Probe update rate based on budget
* depth convolution uses cos(dot(N,RAY_DIR))^25 to weigh ray contributions for visibility, drastically decreasing leaks
* probes are not classified as useful via heuristic but instead are requested by shaded pixels and probe ray hits instead.
* better automatic probe repositioning. PGI also consideres the average free direction and the grids distortion in addition to the closest back and frontface. This is the biggest improvement over the base DDGI in terms of quality. In models like bistro and sponza, the PGI repositioning leaves no visible artifacts while the default DDGI repositioning leaves many "blind spots" causing ugly missinterpolations.
* more robust hysteresis estimation and strong luminance clamping allows PGI to converge faster in challenging scenarios like bistros emissive lights.
* half resolution probe evaluation. The irradiance volume requires 8 probes to be read, that is 8 bisibility and 8 irradiance samples. That is very expensive and even on a RTX4080 leads to up to 0.4ms evaluation time at full resolution. PGI instead evaluates the probes at half res and then uses a depth aware upscale to save performance.
* (TODO) only requested probes are allocated
* (TODO) Cascades allow for much greater view distance
