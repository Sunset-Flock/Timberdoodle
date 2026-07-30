#pragma once

#include <filesystem>
#include <optional>
#include <string>

#include "../scene/tido_format/tido_format.hpp"

auto serialize_tido_metadata_hash(TidoMetadataHash const & hash) -> std::string;
auto serialize_tido_image_descriptor(TidoImageDescriptor const & image) -> std::string;
auto serialize_tido_mesh_descriptor(TidoMeshDescriptor const & mesh) -> std::string;

auto parse_tido_image_header_data(std::span<std::byte const> data) -> std::optional<std::pair<TidoMetadataHash, TidoImageDescriptor>>;
auto parse_tido_mesh_header_data(std::span<std::byte const> data) -> std::optional<std::pair<TidoMetadataHash, TidoMeshDescriptor>>;