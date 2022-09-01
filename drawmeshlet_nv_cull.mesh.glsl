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
  #extension GL_EXT_shader_explicit_arithmetic_types_int8  : require
  #extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#else
  #extension GL_NV_gpu_shader5 : require
  #extension GL_NV_bindless_texture : require
#endif

  #extension GL_KHR_shader_subgroup_basic : require
  #extension GL_KHR_shader_subgroup_ballot : require
  #extension GL_KHR_shader_subgroup_vote : require

//////////////////////////////////////

#include "common.h"

//////////////////////////////////////////////////
// MESH CONFIG

const uint WORKGROUP_SIZE = 32;

layout(local_size_x=WORKGROUP_SIZE) in;
layout(max_vertices=NVMESHLET_VERTEX_COUNT, max_primitives=NVMESHLET_PRIMITIVE_COUNT) out;
layout(triangles) out;

// The workgroup size of the shader may not have enough threads
// to do all the work in a unique thread.
// Therefore we might need to loop to process all the work.
// Looping can have the benefit that we can amortize some registers
// that are common to all threads. However, it may also introduce
// more registers. 

const uint MESHLET_INDICES_ITERATIONS   = ((NVMESHLET_PRIMITIVE_COUNT * 3 + WORKGROUP_SIZE * NVMESHLET_INDICES_PER_FETCH - 1) / (WORKGROUP_SIZE * NVMESHLET_INDICES_PER_FETCH));
const uint MESHLET_VERTEX_ITERATIONS    = ((NVMESHLET_VERTEX_COUNT    + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE);
const uint MESHLET_PRIMITIVE_ITERATIONS = ((NVMESHLET_PRIMITIVE_COUNT + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE);

// task shader is used in advance, doing early cluster culling
#ifndef USE_TASK_STAGE
#define USE_TASK_STAGE          0
#endif

// set in Sample::getShaderPrepend()
// process vertex outputs after primitive culling
// once we know which vertices are actually used
#ifndef USE_VERTEX_CULL
#define USE_VERTEX_CULL    0
#endif

// do frustum culling if primitive culling is active
// otherwise only backface & subpixel is done
#ifndef USE_MESH_FRUSTUMCULL
#define USE_MESH_FRUSTUMCULL    1
#endif

////////////////////////////////////////////////////////////
// optimize configurations

#if USE_TASK_STAGE
// always disable frustumcull on mesh level
// task stage does the heavy lifting
#undef  USE_MESH_FRUSTUMCULL
#define USE_MESH_FRUSTUMCULL 0
#endif

#if SHOW_PRIMIDS || USE_BARYCENTRIC_SHADING
// no attributes exist in these modes, so disable vertex culling
#undef  USE_VERTEX_CULL
#define USE_VERTEX_CULL  0
#endif

/////////////////////////////////////
// UNIFORMS

#if IS_VULKAN

  layout(push_constant) uniform pushConstant{
    // x: mesh, y: prim, z: 0, w: vertex
    uvec4     geometryOffsets;
    // x: meshFirst, y: meshMax
    uvec4     drawRange;
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

  // x: mesh, y: prim, z: 0, w: vertex
  layout(location = 0) uniform uvec4 geometryOffsets;
  // x: meshFirst, y: meshMax
  layout(location = 1) uniform uvec4 drawRange;
  
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
  uint meshletID = gl_WorkGroupID.x + drawRange.x;
#endif
  uint laneID = gl_LocalInvocationID.x;


////////////////////////////////////////////////////////////
// INPUT

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
// VERTEX/PRIMITIVE CULLING SETUP

// When we do per-primitive culling we have two options
// how to deal with the vertex outputs:
// - do them regardless of culling result (USE_VERTEX_CULL == 0)
// - wait until we know which vertices are actually used (USE_VERTEX_CULL == 1)

// NV_mesh_shader allows to have read and write
// access to mesh-shader outputs. Instead of using
// extra shared memory, we simply store temporary
// culling data in the vertex outputs, before we
// later overwrite them with actual outputs.

#if USE_VERTEX_CULL
  void vertexcull_writeVertexIndex(uint vert, uint val)
  {
    OUT[vert].wNormal.x = uintBitsToFloat(val);
  }

  uint vertexcull_readVertexIndex(uint vert)
  {
    return floatBitsToUint(OUT[vert].wNormal.x);
  }

  bool vertexcull_isVertexUsed(uint vert)
  {
    return OUT[vert].wNormal.y != 0;
  }

  void vertexcull_clearVertexUsed(uint vert) {
    OUT[vert].wNormal.y = 0;
  }

  void vertexcull_setVertexUsed(uint vert) {
    OUT[vert].wNormal.y = 1;
  }
  
  // even for primitive culling we hijack
  // the output values
  void primcull_setVertexClip(uint vert, uint mask) {
  #if USE_MESH_FRUSTUMCULL
    OUT[vert].wNormal.z = uintBitsToFloat(mask);
  #endif
  }
  uint primcull_getVertexClip(uint vert) {
    return floatBitsToUint(OUT[vert].wNormal.z);
  }
#else
  // in this scenario we have written vertex outputs already
  // so we cannot repurpose the output space for temporary
  // storage
  
  void primcull_setVertexClip(uint vert, uint mask) {
    // dummy
  }
  uint primcull_getVertexClip(uint vert) {
    return getCullBits(gl_MeshVerticesNV[vert].gl_Position);
  }
#endif

  vec2 primcull_getVertexScreen(uint vert) {
    return getScreenPos(gl_MeshVerticesNV[vert].gl_Position);
  }

//////////////////////////////////////////////////
// VERTEX EXECUTION

// This is the code that is normally done in the vertex-shader
// "vidx" is what gl_VertexIndex would be
//
// We split vertex-shading from attribute-shading,
// to highlight the differences between the drawmeshlet_cull.mesh.glsl
// and drawmeshlet_basic.mesh.glsl files (just use a file-diff
// program to view the two)

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

// One can see that the primary mesh-shader code is agnostic of the vertex-shading work.
// In theory it should be possible to even automatically generate mesh-shader SPIR-V
// as combination of a template mesh-shader and a vertex-shader provided as SPIR-V

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
  {
    UNROLL_LOOP
    for (uint i = 0; i < uint(MESHLET_VERTEX_ITERATIONS); i++)
    {
      uint vert = laneID + i * WORKGROUP_SIZE;
      uint vertLoad = min(vert, vertMax);
      
    #if USE_VERTEX_CULL
      vertexcull_clearVertexUsed(vert);
    #endif

      {
        uint idx   = (vertLoad) >> (vidxDiv-1);
        uint shift = (vertLoad) & (vidxDiv-1);

        uint vidx = primIndices1[idx + vidxStart];
        vidx <<= vidxBits * (1-shift);
        vidx >>= vidxBits;

        vidx += geometryOffsets.w;

        vec4 hPos = procVertex(vert, vidx);
        
        primcull_setVertexClip(vert, getCullBits(hPos));
        
      #if USE_VERTEX_CULL
        // we want to keep the vertex index so that
        // later, after culling, we don't have to load it again
        vertexcull_writeVertexIndex(vert, vidx);
      #else
        // we don't perform vertex culling and directly
        // process the rest of the vertex-shading work here.
        // Otherwise we defer it after triangle culling,
        // so that only the attribute work is done that is
        // really required.
        procAttributes(vert, vidx);  
      #endif
      }
    }
  }

  // PRIMITIVE TOPOLOGY
  {
    // To speed up loading of the primitive (triangle) indices
    // we load 64-bit per thread (NV hardware as fast paths
    // to load aligned 64- and 128-bit values).
    // MESHLET_INDICES_ITERATIONS is typically 1 as result.
    // We also make use of a special intrinsic to distribute
    // the index values into the gl_PrimitiveIndicesNV array.
    //
    // A bit of caution must be taken here, as we must ensure the indices
    // written by the intrinsics fit within the "max_primitives" we
    // declared at start.
    // The nvmeshlet builders actually take care of this and will
    // pack a few less primitives to ensure we never overshoot.
    
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

#else
  #error "NVMESHLET_ENCODING not supported"
#endif

  ////////////////////////////////////////////
  // PRIMITIVE CULLING & OUTPUT PHASE
  
  memoryBarrierShared();
  barrier();
  
  uint outPrimCount = 0;

  UNROLL_LOOP
  for (uint i = 0; i < uint(MESHLET_PRIMITIVE_ITERATIONS); i++)
  {
    uint prim = laneID + i * WORKGROUP_SIZE;
      
    bool   primVisible = false;
    u8vec4 topology;

    if (prim <= primMax) {
      uint idx = prim * 3;
      uint ia = gl_PrimitiveIndicesNV[idx + 0];
      uint ib = gl_PrimitiveIndicesNV[idx + 1];
      uint ic = gl_PrimitiveIndicesNV[idx + 2];
      topology.x = uint8_t(ia);
      topology.y = uint8_t(ib);
      topology.z = uint8_t(ic);
      topology.w = uint8_t(0);

      // build triangle
      vec2 a = primcull_getVertexScreen(ia);
      vec2 b = primcull_getVertexScreen(ib);
      vec2 c = primcull_getVertexScreen(ic);

    #if USE_MESH_FRUSTUMCULL
      // if the task-shader is active and does the frustum culling
      // then we normally don't execute this here
      uint abits = primcull_getVertexClip(ia);
      uint bbits = primcull_getVertexClip(ib);
      uint cbits = primcull_getVertexClip(ic);

      primVisible = testTriangle(a.xy, b.xy, c.xy, 1.0, abits, bbits, cbits);
    #else
      primVisible = testTriangle(a.xy, b.xy, c.xy, 1.0, false);
    #endif

    #if USE_VERTEX_CULL
      if (primVisible) {
        vertexcull_setVertexUsed(ia);
        vertexcull_setVertexUsed(ib);
        vertexcull_setVertexUsed(ic);
      }
    #endif
    }

    uvec4 votePrims = subgroupBallot(primVisible);
    uint  numPrims  = subgroupBallotBitCount(votePrims);
    uint  idxOffset = subgroupBallotExclusiveBitCount(votePrims) + outPrimCount;

    if (primVisible) {
      uint idx = idxOffset * 3;
      gl_PrimitiveIndicesNV[idx + 0] = topology.x;
      gl_PrimitiveIndicesNV[idx + 1] = topology.y;
      gl_PrimitiveIndicesNV[idx + 2] = topology.z;
    #if SHOW_PRIMIDS
      // let's compute some fake unique primitiveID
      gl_MeshPrimitivesNV[idxOffset].gl_PrimitiveID = int((meshletID + geometryOffsets.x) * NVMESHLET_PRIMITIVE_COUNT + prim);
    #endif
    }

    outPrimCount += numPrims;
  }

  ////////////////////////////////////////////  
  // OUTPUT

  memoryBarrierShared();
  barrier();

  if (laneID == 0) {
    gl_PrimitiveCountNV = outPrimCount;
  #if USE_STATS
    atomicAdd(stats.meshletsOutput, 1);
    atomicAdd(stats.trisOutput, outPrimCount);
    #if !USE_VERTEX_CULL
      atomicAdd(stats.attrInput,  vertCount);
      atomicAdd(stats.attrOutput, vertCount);
    #endif
  #endif
  }

#if USE_VERTEX_CULL
  // OUTPUT VERTICES
  {
    uint usedVertices = 0;

    UNROLL_LOOP
    for (uint i = 0; i < uint(MESHLET_VERTEX_ITERATIONS); i++) {
      uint vert = laneID + i * WORKGROUP_SIZE;
      bool used = vertexcull_isVertexUsed( vert );
      if (used) {
        uint vidx = vertexcull_readVertexIndex( vert );
        procAttributes(vert, vidx);
      }
    #if USE_STATS
      uvec4 vote  = subgroupBallot(used);
      uint  verts = subgroupBallotBitCount(vote);
      usedVertices += verts;
    #endif
    }
  #if USE_STATS
    if (laneID == 0){
      atomicAdd(stats.attrInput,  vertCount);
      atomicAdd(stats.attrOutput, usedVertices);
    }
  #endif
  }
#endif
}