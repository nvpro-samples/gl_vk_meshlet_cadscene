/*
 * Copyright (c) 2016-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2016-2022 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */


// ide.config.nvglslcchip="tu100"

#version 450

#ifdef VULKAN
  #extension GL_GOOGLE_include_directive : enable
  #extension GL_EXT_control_flow_attributes: require
  #define UNROLL_LOOP [[unroll]]
#else
  #extension GL_ARB_shading_language_include : enable
  #pragma optionNV(unroll all)
  #define UNROLL_LOOP
#endif

#include "config.h"

//////////////////////////////////////

  #extension GL_NV_mesh_shader : require

//////////////////////////////////////

#if IS_VULKAN
  #extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#else
  #extension GL_NV_gpu_shader5 : require
  #extension GL_NV_bindless_texture : require
#endif


//////////////////////////////////////

#include "common.h"

//////////////////////////////////////////////////
// MESH CONFIG

#define WORKGROUP_SIZE    MESH_SUBGROUP_SIZE

layout(local_size_x=WORKGROUP_SIZE) in;
layout(max_vertices=NVMESHLET_VERTEX_COUNT, max_primitives=NVMESHLET_PRIMITIVE_COUNT) out;
layout(triangles) out;

// task shader is used in advance, doing early cluster culling
#ifndef USE_TASK_STAGE
#define USE_TASK_STAGE          0
#endif

/////////////////////////////////////
// UNIFORMS

#if IS_VULKAN

  layout(push_constant) uniform pushConstant{
    uvec4     geometryOffsets;
    // x: mesh, y: prim, z: 0, w: vertex
  };

  layout(std140, binding = SCENE_UBO_VIEW, set = DSET_SCENE) uniform sceneBuffer {
    SceneData scene;
  };
  layout(std430, binding = SCENE_SSBO_STATS, set = DSET_SCENE) buffer statsBuffer {
    CullStats stats;
  };

  layout(std140, binding= 0, set = DSET_OBJECT) uniform objectBuffer {
    ObjectData object;
  };

  layout(std430, binding = GEOMETRY_SSBO_MESHLETDESC, set = DSET_GEOMETRY) buffer meshletDescBuffer {
    uvec4 meshletDescs[];
  };
  layout(std430, binding = GEOMETRY_SSBO_PRIM, set = DSET_GEOMETRY) buffer primIndexBuffer1 {
    uint  primIndices1[];
  };
  layout(std430, binding = GEOMETRY_SSBO_PRIM, set = DSET_GEOMETRY) buffer primIndexBuffer2 {
    uvec2 primIndices2[];
  };

  layout(binding=GEOMETRY_TEX_VBO,  set=DSET_GEOMETRY)  uniform samplerBuffer  texVbo;
  layout(binding=GEOMETRY_TEX_ABO,  set=DSET_GEOMETRY)  uniform samplerBuffer  texAbo;

#else

  layout(location = 0) uniform uvec4 geometryOffsets;
  // x: mesh, y: prim, z: 0, w: vertex

  layout(std140, binding = UBO_SCENE_VIEW) uniform sceneBuffer {
    SceneData scene;
  };
  layout(std140, binding = SSBO_SCENE_STATS) buffer statsBuffer{
    CullStats stats;
  };

  layout(std140, binding = UBO_OBJECT) uniform objectBuffer {
    ObjectData object;
  };

  // keep in sync with binding order defined via GEOMETRY_
  layout(std140, binding = UBO_GEOMETRY) uniform geometryBuffer{
    uvec4*          meshletDescs;
    uvec2*          primIndices;
    samplerBuffer   texVbo;
    samplerBuffer   texAbo;
  };

  #define primIndices1  ((uint*)primIndices)
  #define primIndices2  primIndices

#endif

/////////////////////////////////////////////////

#include "nvmeshlet_utils.glsl"

/////////////////////////////////////////////////
// MESH INPUT

#if USE_TASK_STAGE
  taskNV in Task {
    uint    baseID;
    uint8_t deltaIDs[NVMESHLET_PER_TASK];
  } IN;
  // gl_WorkGroupID.x runs from [0 .. parentTask.gl_TaskCountNV - 1]
  uint meshletID = IN.baseID + IN.deltaIDs[gl_WorkGroupID.x];
#else
  uint meshletID = gl_WorkGroupID.x;
#endif
  uint laneID = gl_LocalInvocationID.x;


////////////////////////////////////////////////////////////
// INPUT

// We are using simple vertex attributes here, so
// that we can switch easily between fp32 and fp16 to
// investigate impact of vertex bandwith.
//
// In a more performance critical scenario we recommend the use
// of packed normals for CAD, like octant encoding and pack position
// and normal in a single 128-bit value.

// If you work from fixed vertex definitions and don't need dynamic
// format conversions by texture formats, or don't mind
// creating multiple shader permutations, you may want to
// use ssbos here, instead of tbos

vec3 getPosition( uint vidx ){
  return texelFetch(texVbo, int(vidx)).xyz;
}

vec3 getNormal( uint vidx ){
  return texelFetch(texAbo, int(vidx * VERTEX_NORMAL_STRIDE)).xyz;
}

vec4 getExtra( uint vidx, uint xtra ){
  return texelFetch(texAbo, int(vidx * VERTEX_NORMAL_STRIDE + 1 + xtra));
}
  
////////////////////////////////////////////////////////////
// OUTPUT

#if SHOW_PRIMIDS

  // nothing to output

#elif USE_BARYCENTRIC_SHADING

  layout(location=0) out Interpolants {
    flat uint meshletID;
  } OUT[];

  layout(location=1) out ManualInterpolants {
    uint vidx;
  } OUTBary[];

#else

  layout(location=0) out Interpolants {
    vec3  wPos;
    vec3  wNormal;
    flat uint meshletID;
  #if VERTEX_EXTRAS_COUNT
    vec4 xtra[VERTEX_EXTRAS_COUNT];
  #endif
  } OUT[];

#endif

//////////////////////////////////////////////////
// EXECUTION

vec4 procVertex(const uint vert, uint vidx)
{
  vec3 oPos = getPosition(vidx);
  vec3 wPos = (object.worldMatrix  * vec4(oPos,1)).xyz;
  vec4 hPos = (scene.viewProjMatrix * vec4(wPos,1));

  gl_MeshVerticesNV[vert].gl_Position = hPos;

#if !SHOW_PRIMIDS
#if USE_BARYCENTRIC_SHADING
  OUTBary[vert].vidx = vidx;
  OUT[vert].meshletID = meshletID;
#else
  OUT[vert].wPos = wPos;
  OUT[vert].meshletID = meshletID;
#endif
#endif

#if USE_CLIPPING
#if IS_VULKAN
  // spir-v annoyance, doesn't unroll the loop and therefore cannot derive the number of clip distances used
  #if NUM_CLIPPING_PLANES > 0
  gl_MeshVerticesNV[vert].gl_ClipDistance[0] = dot(scene.wClipPlanes[0], vec4(wPos,1));
  #endif
  #if NUM_CLIPPING_PLANES > 1
  gl_MeshVerticesNV[vert].gl_ClipDistance[1] = dot(scene.wClipPlanes[1], vec4(wPos,1));
  #endif
  #if NUM_CLIPPING_PLANES > 2
  gl_MeshVerticesNV[vert].gl_ClipDistance[2] = dot(scene.wClipPlanes[2], vec4(wPos,1));
  #endif
#else
  for (int i = 0; i < NUM_CLIPPING_PLANES; i++){
    gl_MeshVerticesNV[vert].gl_ClipDistance[i] = dot(scene.wClipPlanes[i], vec4(wPos,1));
  }
#endif
#endif

  return hPos;
}


void procAttributes(const uint vert, uint vidx)
{
#if !SHOW_PRIMIDS && !USE_BARYCENTRIC_SHADING
  vec3 oNormal = getNormal(vidx);
  vec3 wNormal = mat3(object.worldMatrixIT) * oNormal;
  OUT[vert].wNormal = wNormal;
  #if VERTEX_EXTRAS_COUNT
    UNROLL_LOOP
    for (int i = 0; i < VERTEX_EXTRAS_COUNT; i++) {
      vec4 xtra = getExtra(vidx, i);
      OUT[vert].xtra[i] = xtra;
    }
  #endif
#endif
}

//////////////////////////////////////////////////
// MESH EXECUTION

  // The workgroup size of the shader may not have enough threads
  // to do all the work in a unique thread.
  // Therefore we might need to loop to process all the work.
  // Looping can have the benefit that we can amortize some registers
  // that are common to all threads. However, it may also introduce
  // more registers. 
  
  #define MESHLET_INDICES_ITERATIONS    ((NVMESHLET_PRIMITIVE_COUNT * 3 + WORKGROUP_SIZE * NVMESHLET_INDICES_PER_FETCH - 1) / (WORKGROUP_SIZE * NVMESHLET_INDICES_PER_FETCH))
  #define MESHLET_VERTEX_ITERATIONS     ((NVMESHLET_VERTEX_COUNT    + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
  #define MESHLET_PRIMITIVE_ITERATIONS  ((NVMESHLET_PRIMITIVE_COUNT + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)

///////////////////////////////////////////////////////////////////////////////

void main()
{

#if NVMESHLET_ENCODING == NVMESHLET_ENCODING_PACKBASIC

  // LOAD HEADER PHASE
  uvec4 desc = meshletDescs[meshletID + geometryOffsets.x];

  uint vertMax;
  uint primMax;

  uint vidxStart;
  uint vidxBits;
  uint vidxDiv;
  uint primStart;
  uint primDiv;

  decodeMeshlet(desc, vertMax, primMax, primStart, primDiv, vidxStart, vidxBits, vidxDiv);

  vidxStart += geometryOffsets.y / 4;
  primStart += geometryOffsets.y / 4;

  uint primCount = primMax + 1;
  uint vertCount = vertMax + 1;

  // VERTEX PROCESSING
  
  UNROLL_LOOP
  for (uint i = 0; i < uint(MESHLET_VERTEX_ITERATIONS); i++)
  {
    uint vert = laneID + i * WORKGROUP_SIZE;
    uint vertLoad = min(vert, vertMax);

    {
      uint idx   = (vertLoad) >> (vidxDiv-1);
      uint shift = (vertLoad) &  (vidxDiv-1);

      uint vidx = primIndices1[idx + vidxStart];
      vidx <<= vidxBits * (1-shift);
      vidx >>= vidxBits;

      vidx += geometryOffsets.w;

      procVertex(vert, vidx);
      procAttributes(vert, vidx);
    }
  }

  // PRIMITIVE TOPOLOGY
  {
    uint readBegin = primStart / 2;
    uint readIndex = primCount * 3 - 1;
    uint readMax   = readIndex / 8;

    UNROLL_LOOP
    for (uint i = 0; i < uint(MESHLET_INDICES_ITERATIONS); i++)
    {
      uint read = laneID + i * WORKGROUP_SIZE;
      uint readUsed = min(read, readMax);
      uvec2 topology = primIndices2[readBegin + readUsed];
      writePackedPrimitiveIndices4x8NV(readUsed * 8 + 0, topology.x);
      writePackedPrimitiveIndices4x8NV(readUsed * 8 + 4, topology.y);
    }
  }
  #if SHOW_PRIMIDS
  {
    UNROLL_LOOP
    for (uint i = 0; i < uint(MESHLET_PRIMITIVE_ITERATIONS); i++)
    {
      uint prim = laneID + i * WORKGROUP_SIZE;
      if (prim <= primMax) {
        gl_MeshPrimitivesNV[prim].gl_PrimitiveID = int((meshletID + geometryOffsets.x) * NVMESHLET_PRIMITIVE_COUNT + prim);
      }
    }
  }
  #endif

#else
  #error "NVMESHLET_ENCODING not supported"
#endif

  ////////////////////////////////////////////

  if (laneID == 0) {
    gl_PrimitiveCountNV = primCount;
  #if USE_STATS
    atomicAdd(stats.meshletsOutput, 1);
    atomicAdd(stats.trisOutput, primCount);
    atomicAdd(stats.attrInput,  vertCount);
    atomicAdd(stats.attrOutput, vertCount);
  #endif
  }
}
