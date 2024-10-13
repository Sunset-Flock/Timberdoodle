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

### Debug views (Matěj Sakmary & Patrick Ahrens)
As we believe debug views are very important and helpful when idetifying and fixing bugs, Timberdoodle includes several drawing modes visualizing so far implemented techniques. As we fully adopted meshlet workflow to allow for efficient culling, we include a meshlet, mesh, entity and triangle views. To visualize the effects of culling we include overdraw visualizations for both main pass as well as shadowmaps. For Shadowmaps an additional debug view (shown at the top of the Virtual Shadow Maps section) is also included. This debug view visualizes individual cascades and pages. Another very important feature we implemented is the observer camera. This feature allows to freeze the main camera and detach another, observer, camera which can freely move around the scene and inspect its state as if it has been drawn from the main camera. This option was priceless when debugging various artifacts which were hard to argue about from the point of view of the main camera but were immediatelly made clear when inspected using the observer.

Additionally to debug views Timberdoodle also includes additional set of debug textures used for various purposes. One example is the presence of textures showing the state of the VSM memory pool as well as the state of each of the cascade VPTs. To further allow for easy debugging, the option do draw simple shapes such as rectangles, AABBs, cubes, spheres and other are built into the core of the engine. The debug shape drawing is supported both from the CPU as well as the GPU. This allows us to immediatelly visualize wide spectrum of data and quickly gain deeper insight into potential issues.

![](https://github.com/Sunset-Flock/Timberdoodle/blob/main/media/overdraw_bistro.png)
![](https://github.com/Sunset-Flock/Timberdoodle/blob/main/media/meshlet_bistro.png)
![](https://github.com/Sunset-Flock/Timberdoodle/blob/main/media/culling_bistro.png)

### Two Pass Culling Pipeline (Patrick Ahrens)

Tido uses a visibility buffer based two pass culling. This has several advantages over more conventional two pass culling. To show the difference i ll make a birds eye description of how tido performs two pass culling and how it is conventionally done:

Conventional two-pass culling:
* Step 1: Build hierarchical depth image (hiz) from last frames depth image
* Step 2: Cull all mesh instances
* Step 3: Cull meshlet instances of non-culled mesh instances
* Step 4. Draw non-culled meshlet instances
* Step 5: Build hierarchical depth image (hiz) for current new depth image
* Step 6: Cull all mesh instances
* Step 7: Cull meshlet instances of non-culled mesh instances
* Step 8. Draw non-culled meshlet instances

And here is how tido performs two pass culling using the visibility buffer:

* Step 1: Analyze triangle id image of last frame, build list of visible meshlets
* Step 2: Build meshlet bitfield, marking all meshlets drawn in first pass (used to avoid drawing these meshlets a second time in the second pass)
* Step 3: Draw triangle id and depth for all visible meshlets from the previous frame.
* Step 4: Build hierarchical z buffer (hiz) from the drawn depth map.
* Step 5: Cull all mesh instances against frustum and hiz
* Step 6: Cull meshlets of non-culled meshes against frustum and hiz
* Step 7: Draw triangle id and depth of all non-culled meshlets

Steps 4-8 in the conventional way and Steps 3-7 in the tido way are very similar.
But Steps 1-3 and 1-2 in tido are very different. Instead of culling all meshles and meshlets, tido uses the tri id image from the last frame, to build a list of visible meshlets. 
These pre steps are quite different but take a similar amount of time on the gpu. 

The first benefit of the tido approach is, that the amount of geo drawn in the first pass is much lower. This is because hiz culling is necessarily conservative and coarse. It will always have many false negatives, drawing more geo than is actually visible. Using the triangle id image has the big advantage of giving you the exact meshlets that are visible. This typically reduces the meshlet count drawn by 40-70%, leading to a speedup of 30-60% for the first pass draws!

The second big benefit arises from two smaller changes to the analyze triangle id image pass. Firstly, move it from `Step 1` to `Step 7`. This gives us the list of visible meshlets not only of the last but also the current frame. Secondly, also write out the visible triangles in the analyze triangle id image pass.

It might be surprising, due to the high spacial coherence, writing out hte visible triangles is actually very cheap and does not change the runtime much at all.

Now with this we can perform perfectly culled forward shading! Forward shading is still the fastest way to draw when we arent quad occupancy limited AND/OR the vertex shader is much more expensive than the pixel shader (this is the case for example when drawing motion vectors for animated geo).

Other details:
All meshlet culling is performed by task/amplification shaders in the mesh shading pipeline. Aside from meshlet culling, tido also does fine grained triangle culling. This is especially effective for very large triangles.