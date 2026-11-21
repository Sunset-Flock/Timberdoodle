#pragma once

#include "pgi_update.inl"

// CPU state of the PGI probe inspection (UI: PGI Settings -> Probe Inspection).
struct PGIInspectState
{
    // Clicking a probe in the viewport selects it, the inspection pass runs for the selected probe.
    bool enabled = false;
    bool show_rays = true;
    bool has_selection = false;
    // Selected probe: absolute probe grid coordinate xyz (window index + window offset - count / 2 + 1, stable while the window scrolls) and cascade.
    daxa_i32 selected_grid_coord[4] = {};
    // Host visible data the inspection pass writes every frame (null until the first graph was built).
    PGIInspectData const * data = nullptr;
};
