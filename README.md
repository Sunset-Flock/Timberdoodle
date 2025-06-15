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

![](https://github.com/Sunset-Flock/Timberdoodle/blob/main/media/visbuffer_meshlets.png)

Tido uses a gpu-driven, fully bindless rendering pipeline. All geometry is mesh, then cluser, then triangle culled and finally drawn to a visbuffer (triangle id + depth).

#### Mesh Preparation

When loading assets, tido preprocesses all meshes:
1. optimize vertices of each mesh
2. auto generate lods for each mesh
    - each lod is roughly half the triangle count of the previous lod
    - uses error metric based on normal and positional error for each simplification
    - up to 16 lods are generated, the generation of lods is stopped after crossing an error threshold
    - generate culling data (aabb, bounding sphere)
3. generate meshlets for each mesh
    - up to 64 triangles and 64 vertices per meshlet
    - compressed indexbuffer for each meshlet
    - generate culling data (aabb, bounding sphere)

Most of these operations use the meshoptimizer library

Each mesh with all its data (indices, vertices meshlets ...) is packed into a single buffer allocation. A GPUMesh struct is created holding metadata (vertex count, aabb...) and pointers to the section of the mesh buffer containing lists of all the meshes attributes (indices, meshlets, vertices...).

> In tido we exclusively use Buffer Device Address pointers and never buffer objects in shader code

This GPUMesh struct is then put into a buffer (called mesh manifest) containing an array with all GPUMeshes. This buffer is available in all shaders in shading, allowing any rendering shader to always access all geometry data.

#### Mesh Data Structures

In tido, each object/ render entity can have a single MeshGroup. Multiple RenderEntities can share these MeshGroups.
A Meshgroup consists of multiple MeshLodGroups. A MeshGroup is a collection of multiple meshes with different materials/drawing requirements.
A MeshLodGroup is a Mesh and all its Lods, each MeshLodGroup holds multiple Meshes (the lods of the MeshLodGroup). 

![](https://github.com/Sunset-Flock/Timberdoodle/blob/main/media/mesh_data_structures.png)

#### CPU Side Drawing / Lod Selection

An example of an artificial stress test scene for culling/lodding.
This scene contains 8192 bunnies with each 1082 meshlets and 64,920 triangled for a total of 531,694,800 triangles.
Tido renders the scene in around 250-800us with culling and lodding enabled. With no culling and lod0, this scene takes over 121ms to render on my RTX4080.

![](https://github.com/Sunset-Flock/Timberdoodle/blob/main/media/bunnies2_lod.png)
![](https://github.com/Sunset-Flock/Timberdoodle/blob/main/media/bunnies2_meshlets.png)

At the beginning of each frame, tido goes over all render entities and their mesh groups in the scene and selects an appropriate lod for each of the MeshLodGroups on the cpu.
The lod is selected based on a rough pixel error metric. Each each lod's bounding box is projected into a rotated view towards the lod. Rotating the view for the projection ensures that there is never an lod change based on the real cameras orientation, as well as making the selection consistent with raytracing.

The projected bounding boxes pixel size is multiplied with the estimated error calculated by the mesh lod generation. The lod with the highest error that still falls under ~1 pixel error is selected.
This scheme is usually good enough to ensure that lod changes are not visible.

```c
struct MeshInstance
{
    daxa_u32 entity_index;
    daxa_u32 mesh_index;
    daxa_u32 in_mesh_group_index;
    daxa_u32 mesh_group_index;
    daxa_u32 flags;
};
```
After the lod is selected, a `MeshInstance` is pushed into a per frame list of mesh instances. 

The mesh instance is then also added to draw lists. Drawlists are used to filter geometry into different passes, so for example, different pipeline permutations for Visbuffer drawing has different drawlists, shadow map draws have their own drawlist and other things like cache invalidation (moving objects invalidating static shadow cache) have their own drawlist. 

A drawlist entry is just a index into the list of MeshInstances.

The gpu then only processes the mesh instances and drawlists directly. It never has to interact or resolve the indirections from render `entity -> mesh group -> mesh lod group -> mesh`, it only ever sees mesh instances.

For rasterization, this is all the cpu does for individual mesh processing, all commands recorded to draw the mesh instances later are batch processing mesh instances all at once. No individual drawcalls/dispatches instances of anything.

The generated drawlists are also used to generate a list of blases for the tlas of the raytracing acceleration structure. 

The flattening of indirections and acceleration structure builds requireing cpu calls are the main reasons tido performs lod selection on the cpu currently.

But this structure is also generally very convenient as its trivial to exclude/ add/ mark meshes and mesh instances on the cpu this way while having near all of the benefit of full gpu driven rendering. 

#### Draw Pass / Mega Drawcall

Reguardless of mesh instance count, tido can draw all geometry "in one drawcall". This means extremely low cpu overhead for any amount of geometry on the screen.
This way of drawing geometry also reduces gpu state switches to a minimum. That yields great performance, especially for very large triangle and instance counts.

To achieve this, tido must perform multiple phases of "work expansion":
1. a compute shader goes over all mesh instances
    - culls meshes (described later)
    - enqueues the surviving meshes meshlets into a "work expansion" buffer
2. a indirect compute shader dispatch goes over all surviving meshlets using the "work expansion" buffer
    - culls meshlets 
    - writes surviving meshlets into a meshlet instance buffer 
3. a indirect mesh shader dispatch goes over all meshlet instances and draws them.
    - uses the prepared meshlets
    - draws visibility id and depth
    - visibility id is 24 bit meshlet instance index and 6 bits triangle index 2 bit to spare
    - uses inverse z to get best precision and infinite far plane
    - as all geometry is bindless, the indirections from meshlet_instance->mesh_instance->mesh allows this to be one drawcall

The second step can also be performed by task shaders instead of a compute pass. Sadly, task shaders have poor performance for very high meshlet counts. This makes them slower on my gpu (RTX4080) for high meshlet counts. For lowish meshlet counts (<100k) the task shaders are faster tho because they avoid the barrier between compute and mesh shader as well as allow for some overlap between meshshading and meshlet culling work. 

> TIDO now uses async compute to overlap a few misc work passes (Tlas build, sky generation, light culling) with the beginning of the frame, hiding most of the latency of a separate compute cull. This makes the task shader path even less useful as async compute can overlap the induced barrier very well.

There is a catch to the "one drawcall" to draw everything, as the Pixelshader for alpha discard geometry prevents early z tests due to the discard call. To prevent general performance degradation, Tido has two rasterization pipelines, one without alpha discard that can perform early z tests used on opaque. And one pipeline that does use discard to support alpha discard geometry.

This "work expansion" step is using a unique strategy (at least i believe it is, i have yet to find anyone else do this) that i call "power of two expansion". 
This approach tries to minimize the execution cost of adding workitems for a following pass (avoids potential long loops in inserting indirect arguments) while also keeping unpacking of the workitems in the expanded threads to a minimum.
With taskshaders, this approach is a lot faster (up to 25%) in many cases compared to something like prefix sum + binary search, as the work unpacking step is much quicker.
[More on that here](https://gist.github.com/Ipotrick/d6bcf9696b00c72eb037370794733de8)

References: 
* https://developer.nvidia.com/blog/introduction-turing-mesh-shaders/
* https://gist.github.com/Ipotrick/d6bcf9696b00c72eb037370794733de8
* https://github.com/zeux/meshoptimizer
* http://filmicworlds.com/blog/visibility-buffer-rendering-with-material-graphs/

#### GPU Culling

An illustration of the culling. The first image shows the view of the main camera. The second image shows the scene post-culling from an observer camera for that same main camera view.

![](https://github.com/Sunset-Flock/Timberdoodle/blob/main/media/visbuffer_cull_main_camera.png)
![](https://github.com/Sunset-Flock/Timberdoodle/blob/main/media/visbuffer_culled.png)

Tido performs all culling on the gpu. 

Tido culls against frustum, hiz (hierarchical z buffer), for meshes and meshlets. For triangles, tido additionally culls backfaces and triangles that miss all rasterization samples.
Culling backfaces in the mesh shader instead of letting the hardware cull them can be significantly faster on some HW. It also lets us avoid another state switch.

First, tido builds an HIZ of the previous frames depth buffer. The HIZ builder is a optimized single pass downsampler (takes ~34us on a RTX4080 for a 1440p depth image).
It then culls meshes, meshlets in their position of the last frame against the previous frames HIZ.
It then draws the culled meshlets in their current frame positions to the visbuffer.

This way we can effectively reuse the visibility information the depth buffer gives us from the last frame, saving us a lot of performance in drawing.

There is one issue with this: Due to camera or object movement, the last frames depth is outdated. This means we might cull things, that are actually visible in the current frame!

This is why TIDO performs a second pass, culling and rendering the visbuffer a second time.
A new HIZ is generated of the partially complete visbuffer from the first pass.
Meshes and meshlets are culled against it and the frustum and then drawn to the visbuffer.

The resulting visbuffer after the second pass is complete, as the new HIZ we use for culling is using information from the current frame, it will not overcull.

Without further change, this second pass will redraw most of the geometry that was already drawin in the first pass.
Because the mesh and meshlet culling has to be quick and conservative it can not detect that the meshlets were already drawn in the first pass and will let be drawn a second time.
To avoid this, tido builds a meshlet bitfileld every frame, noting which meshlets were drawn in the first pass, that is read in the second pass to early out for any meshlet that was already drawn.

In the mesh cull phase a bitfield section is allocated for each passing mesh. It will allocate a bit per meshlet in the surviving mesh.
In the meshlet cull phase, the bits in the bitfield are then set in the first pass, when the meshlet passes the culling.
In the second phase meshlet cull phase this bitfield is read in again and meshlets that were already drawn in the first pass early out of the culling.

In practive this drastically improves performance. The early out in the second meshlet cull phase massively reduces the time taken for culling and especially the drawing time for the second pass improves alot.

An alternative way of doing the two pass drawing would be to output a list of culled meshlets in the first pass to then be culled again in the second pass. 
The issue with this approach is that the number of culled meshlets can be gigantic compared to the number of actually drawn meshlets. 
The As this is all gpu driven, the buffers are all pre allocated and can now grow while the gpu executes. So tido must already allocate quite large worse case fixed size buffers for the visbuffer pipeline. 
Adding another buffer with worst case size for all culled meshlets would be a massive VRAM use increase. 
I have also tested the performance difference to what tido does and its quite minimal in most cases. The mesh and meshlet cull both are very fast and the early out in the meshlet cull helps a lot.

#### Visbuffer

Just using the visbuffer, we can fully reconstruct all triangle information in a deferred manner when shading later. 

All geometry data in tido is bindless. This allows tido to fetch any information about any triangle in the visbuffer in any shader.
No need for multiple passes or thousands of bind calls to be able to shader the visbuffer, its all directly available.

To decode the visbuffer:
- tido fetches the triangle id
- splits it into meshlet instance id and meshlet triangle id
- fetches the meshlet instance, this gives us the mesh intance id
- fetches the meshes meta data, which gives us a pointer to the meshlet, vertex and triangle data
- fetches the three vertices of the meshlets triangle
- transforms them and their attributes
- calculates bayecentrics based on the vertex positions and pixel primary ray
- interpolates the vertices using the barycentrics

The visbuffer is decoded twice. Once, to generate a slim gbuffer that is required for deferred effects such as RTAO, PGI and RT Reflections.
The second decoding is when shading opaque geometry. 
Decoding it twice gives us two advantages: The gbuffer can be much slimmer as it does not have to be complete, and we have access to all data of a triangle, without it having to go through the gbuffer first for shading later.
This is very convenient when prototying new shading features as we do not have to modify the gbuffer at all, the visbuffer gives us everything in the shading pass.

References: 
* http://filmicworlds.com/blog/visibility-buffer-rendering-with-material-graphs/
* https://research.activision.com/publications/2021/09/geometry-rendering-pipeline-architecture

### Probe Based Global Illumination (PGI) (Patrick Ahrens)

![](https://github.com/Sunset-Flock/Timberdoodle/blob/main/media/shaded.png)
![](https://github.com/Sunset-Flock/Timberdoodle/blob/main/media/probes.png)
![](https://github.com/Sunset-Flock/Timberdoodle/blob/main/media/pgi_cascades.png)

Based on [RTXGI-DDGI](https://github.com/NVIDIAGameWorks/RTXGI-DDGI).

The basic idea is to cover the world in an irradiance volume made up of world space probes. 
Rays are shot from each probe in real time, calculating a depth and irradiance map for each probe.
Then, any point in the world can be shaded by fetching the 8 closest probes, testing for visibility and interpolating their irradiance.

PGI in timberdoodle makes a few improvements over the base DDGI algorithm:
* Variable Probe update rate: Instead of tracing rays for every probe each frame, tido only updates parts of the probes each frame, default is 1 in 8.
* Tighter depth test: Depth convolution uses cos(dot(N,RAY_DIR))^50 to weigh ray contributions for visibility, drastically reducing leaks.
* Sparsity: Probes are requested from shaded screen pixels and probe trace hits. Allows for one indirection for requests screen pixel -> probe -> probe. This reduces the number of active probes by orders of magnitude. 
* Smarter probe repositioning: PGI also consideres the average free direction and the grids distortion in addition to the closest back and frontface. This is the biggest improvement over the base DDGI in terms of quality. In models like bistro and sponza, the PGI repositioning leaves no visible artifacts while the default DDGI repositioning leaves many "blind spots" causing ugly missinterpolations.
* Robuster hysteresis estimation: Allows PGI to converge faster in challenging scenarios like bistros emissive lights.
* Cascades/Clips: Tido generates multiple cascades of probes around the player, each beeing 2x the size of the previous. Per default each cascade has at most 32^3 probes in 8 cascades.
