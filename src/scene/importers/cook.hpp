#pragma once

#include <memory>
#include <variant>

#include "../../timberdoodle.hpp"
#include "../../multithreading/thread_pool.hpp"
#include "../importer_types.hpp"
#include "../streamer.hpp"
using namespace tido::types;

// Identifies an entry inside of owning Importer source.
struct SourceIndentifier
{
    // Index of the source inside the Importer's list of sources.
    u32 source_index = {};
    // The index of the slot inside the source's list of slots.
    u32 slot_index = {};
    // The generation of the source.
    u32 generation = {};
};

struct CookTask : Task
{
    SourceIndentifier identifier = {};
    // A cook that produced no artifact leaves the monostate it started in, so failure needs no separate flag.
    std::variant<std::monostate, ImageStreamerData, MeshStreamerData> streamer_data = {};
};

/// --- Cooks ---
/// One asset produced out of resolved importer data:
//      - hash the sources
//      - derive the artifact key from that hash
//      - and either serve the artifact already sitting at the key or cook and write one.
auto cook_image(ImageImporterData importer_data, SourceIndentifier slot) -> std::shared_ptr<CookTask>;
auto cook_mesh(MeshImporterData importer_data, SourceIndentifier slot) -> std::shared_ptr<CookTask>;
auto cook_vdb(VdbImporterData importer_data, SourceIndentifier slot) -> std::shared_ptr<CookTask>;
