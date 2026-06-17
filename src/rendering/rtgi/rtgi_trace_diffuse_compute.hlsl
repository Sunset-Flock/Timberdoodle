#pragma once

#define DAXA_RAY_TRACING 1

#include <daxa/daxa.inl>

#include "rtgi_trace_diffuse.hlsl"
#include "rtgi_trace_diffuse_shading.hlsl"

struct ComputePipelineTraceRay : TraceRayInterface
{
    static void trace_and_shade(RayDesc ray, uint flags, inout RayPayload payload)
    {
        let push = rtgi_trace_diffuse_push;

        RayQuery<RAY_FLAG_NONE> q;
        q.TraceRayInline(
            RaytracingAccelerationStructure::get(push.attach.tlas),
            flags,
            ~0,
            ray
        );

        // Proceed loop handles non-opaque (any-hit) candidate intersections.
        while (q.Proceed())
        {
            if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
            {
                if (rt_is_alpha_hit(
                    push.attach.globals,
                    push.attach.mesh_instances,
                    push.attach.globals.scene.meshes,
                    push.attach.globals.scene.materials,
                    q.CandidateTriangleBarycentrics(),
                    q.CandidatePrimitiveIndex(),
                    q.CandidateInstanceID(),
                    ray.Origin, ray.Direction, q.CandidateTriangleRayT()))
                {
                    q.CommitNonOpaqueTriangleHit();
                }
            }
        }

        if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
        {
            shade_closest_hit(payload,
                q.CommittedTriangleBarycentrics(),
                q.CommittedPrimitiveIndex(),
                q.CommittedGeometryIndex(),
                q.CommittedInstanceID(),
                ray.Origin, ray.Direction, q.CommittedRayT());
        }
        else
        {
            shade_miss(payload, ray.Origin, ray.Direction, ray.TMax);
        }
    }
}

[shader("compute")]
[numthreads(8,8,1)]
void ray_gen_compute(uint2 dtid : SV_DispatchThreadID)
{
    shade_ray_gen<ComputePipelineTraceRay>(dtid);
}