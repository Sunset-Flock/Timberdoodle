#pragma once

#include <span>

#include "../../timberdoodle.hpp"
using namespace tido::types;

constexpr f32vec3 DEFAULT_UNPACK_DOT = f32vec3(0.96414679f, 0.03518212f, 0.00067109f);

void CompressBlockBC1SDF(u64 *outBCBlock, std::span<float> inInputData, f32vec3 inUnpackDot = DEFAULT_UNPACK_DOT);
void DecompressBlockBC1SDF(std::span<float> outValues, u64 const * inBCBlock, f32vec3 inUnpackDot = DEFAULT_UNPACK_DOT);