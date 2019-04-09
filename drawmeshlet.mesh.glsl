/* Copyright (c) 2016-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// ide.config.nvglslcchip="tu100"

#version 450

#extension GL_GOOGLE_include_directive : enable
#extension GL_ARB_shading_language_include : enable

#include "config.h"

//////////////////////////////////////

  #extension GL_NV_mesh_shader : require

//////////////////////////////////////

#if IS_VULKAN
  // one of them provides uint8_t
  #extension GL_KHX_shader_explicit_arithmetic_types_int8 : enable
  #extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
  #extension GL_NV_gpu_shader5 : enable
    
  #extension GL_KHR_shader_subgroup_basic : require
  #extension GL_KHR_shader_subgroup_ballot : require
  #extension GL_KHR_shader_subgroup_vote : require
#else
  #extension GL_NV_gpu_shader5 : require
  #extension GL_NV_bindless_texture : require
  #extension GL_NV_shader_thread_group : require
  #extension GL_NV_shader_thread_shuffle : require
#endif

//////////////////////////////////////

#include "common.h"

//////////////////////////////////////////////////
// MESH CONFIG

#define GROUP_SIZE    WARP_SIZE

layout(local_size_x=GROUP_SIZE) in;
layout(max_vertices=NVMESHLET_VERTEX_COUNT, max_primitives=NVMESHLET_PRIMITIVE_COUNT) out;
layout(triangles) out;

// do primitive culling in the shader, output reduced amount of primitives
#ifndef USE_MESH_SHADERCULL
#define USE_MESH_SHADERCULL     0
#endif

// no cull-before-fetch, always load all attributes
#ifndef USE_EARLY_ATTRIBUTES
#define USE_EARLY_ATTRIBUTES    0
#endif

// task shader is used in advance, doing early cluster culling
#ifndef USE_TASK_STAGE
#define USE_TASK_STAGE          0
#endif

// do frustum culling if USE_MESH_SHADERCULL is active
// otherwise only backface & subpixel is done
#ifndef USE_MESH_FRUSTUMCULL
#define USE_MESH_FRUSTUMCULL    1
#endif

// get compiler to do batch loads
#ifndef USE_BATCHED_LATE_FETCH
#define USE_BATCHED_LATE_FETCH  1
#endif

////////////////////////////////////////////////////////////
// optimize configurations

#if !USE_MESH_SHADERCULL
// always must use early attributes if culling is disabled
#undef  USE_EARLY_ATTRIBUTES
#define USE_EARLY_ATTRIBUTES    1
#endif

#if USE_TASK_STAGE
// always disable frustumcull on mesh level
// task stage does the heavy lifting
#undef  USE_MESH_FRUSTUMCULL
#define USE_MESH_FRUSTUMCULL 0
#endif

/////////////////////////////////////
// UNIFORMS

#if IS_VULKAN

  #if USE_PER_GEOMETRY_VIEWS
    uvec4 geometryOffsets = uvec4(0, 0, 0, 0);
  #else
    layout(push_constant) uniform pushConstant{
      uvec4     geometryOffsets;
    };
  #endif

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
  layout(std430, binding = GEOMETRY_SSBO_PRIM, set = DSET_GEOMETRY) buffer primIndexBuffer {
    uvec2 primIndices[];
  };

  layout(binding=GEOMETRY_TEX_IBO,  set=DSET_GEOMETRY)  uniform usamplerBuffer texIbo;
  layout(binding=GEOMETRY_TEX_VBO,  set=DSET_GEOMETRY)  uniform samplerBuffer  texVbo;
  layout(binding=GEOMETRY_TEX_ABO,  set=DSET_GEOMETRY)  uniform samplerBuffer  texAbo;

#else

  #if USE_PER_GEOMETRY_VIEWS
    uvec4 geometryOffsets = uvec4(0,0,0,0);
  #else
    layout(location = 0) uniform uvec4 geometryOffsets;
    // x: mesh, y: prim, z: index, w: vertex
  #endif

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
    usamplerBuffer  texIbo;
    samplerBuffer   texVbo;
    samplerBuffer   texAbo;
  };
  
#endif

/////////////////////////////////////////////////

#include "nvmeshlet_utils.glsl"

/////////////////////////////////////////////////
// MESH INPUT

#if USE_TASK_STAGE
  taskNV in Task {
    uint    baseID;
    uint8_t subIDs[GROUP_SIZE];
  } IN;
  // gl_WorkGroupID.x runs from [0 .. parentTask.gl_TaskCountNV - 1]
  uint meshletID = IN.baseID + IN.subIDs[gl_WorkGroupID.x];
#else
  uint meshletID = gl_WorkGroupID.x;
#endif
  uint laneID = gl_LocalInvocationID.x;


////////////////////////////////////////////////////////////
// INPUT

// If you work from fixed vertex definitions and don't need dynamic 
// format conversions by texture formats, or don't mind
// creating multiple shader permutations, you may want to
// use ssbos here, instead of tbos for a bit more performance.

vec3 getPosition( uint vidx ){
  return texelFetch(texVbo, int(vidx)).xyz;
}

vec3 getNormal( uint vidx ){
  return texelFetch(texAbo, int(vidx * NORMAL_STRIDE)).xyz;
}

vec4 getExtra( uint vidx, uint xtra ){
  return texelFetch(texAbo, int(vidx * NORMAL_STRIDE + 1 + xtra));
}

////////////////////////////////////////////////////////////
// OUTPUT

layout(location=0) out Interpolants {
  vec3  wPos;
  float dummy;
  vec3  wNormal;
  flat uint meshletID;
#if EXTRA_ATTRIBUTES
  vec4 xtra[EXTRA_ATTRIBUTES];
#endif
} OUT[];

//////////////////////////////////////////////////
// EXECUTION

vec4 procVertex(const uint vert, uint vidx)
{
  vec3 oPos = getPosition(vidx);
  vec3 wPos = (object.worldMatrix  * vec4(oPos,1)).xyz;
  vec4 hPos = (scene.viewProjMatrix * vec4(wPos,1));
  
  gl_MeshVerticesNV[vert].gl_Position = hPos;
  
  OUT[vert].wPos = wPos;
  OUT[vert].dummy = 0;
  OUT[vert].meshletID = meshletID;
  
#if USE_CLIPPING
#if IS_VULKAN
  // spir-v annoyance, doesn't unroll the loop and therefore cannot derive the number of clip distances used
  gl_MeshVerticesNV[vert].gl_ClipDistance[0] = dot(scene.wClipPlanes[0], vec4(wPos,1));
  gl_MeshVerticesNV[vert].gl_ClipDistance[1] = dot(scene.wClipPlanes[1], vec4(wPos,1));
  gl_MeshVerticesNV[vert].gl_ClipDistance[2] = dot(scene.wClipPlanes[2], vec4(wPos,1));
#else
  for (int i = 0; i < NUM_CLIPPING_PLANES; i++){
    gl_MeshVerticesNV[vert].gl_ClipDistance[i] = dot(scene.wClipPlanes[i], vec4(wPos,1));
  }
#endif
#endif
  
  return hPos;
}

// To benefit from batched loading, and reduce latency 
// let's make use of a dedicated load phase.
// (explained at the end of the file in the USE_BATCHED_LATE_FETCH section)

#if USE_BATCHED_LATE_FETCH
  struct TempAttributes {
    vec3 normal;
  #if EXTRA_ATTRIBUTES
    vec4 xtra[EXTRA_ATTRIBUTES];
  #endif
  };

  void fetchAttributes(inout TempAttributes temp, uint vert, uint vidx)
  {
    temp.normal = getNormal(vidx);
  #if EXTRA_ATTRIBUTES
    for (int i = 0; i < EXTRA_ATTRIBUTES; i++){
      temp.xtra[i] = getExtra(vidx, i);
    }
  #endif
  }

  void storeAttributes(inout TempAttributes temp, const uint vert, uint vidx)
  {
    vec3 oNormal = temp.normal;
    vec3 wNormal = mat3(object.worldMatrixIT) * oNormal;
    OUT[vert].wNormal = wNormal;
  #if EXTRA_ATTRIBUTES
    for (int i = 0; i < EXTRA_ATTRIBUTES; i++){
      OUT[vert].xtra[i] = temp.xtra[i];
    }
  #endif
  }

  void procAttributes(const uint vert, uint vidx)
  {
    TempAttributes  temp;
    fetchAttributes(temp, vert, vidx);
    storeAttributes(temp, vert, vidx);
  }

#else
  
  // if you never intend to use the above mechanism,
  // you can express the attribute processing more like a regular
  // vertex shader

  void procAttributes(const uint vert, uint vidx)
  {
    vec3 oNormal = getNormal(vidx);
    vec3 wNormal = mat3(object.worldMatrixIT) * oNormal;
    OUT[vert].wNormal = wNormal;
  #if EXTRA_ATTRIBUTES
    for (int i = 0; i < EXTRA_ATTRIBUTES; i++) {
      vec4 xtra = getExtra(vidx, i);
      OUT[vert].xtra[i] = xtra;
    }
  #endif
  }
#endif

//////////////////////////////////////////////////
// MESH EXECUTION

#if !USE_EARLY_ATTRIBUTES
  void writeVertexIndex(uint vert, uint val)
  {
    OUT[vert].wNormal.x = uintBitsToFloat(val);
  }
  
  uint readVertexIndex(uint vert)
  {
    return floatBitsToUint(OUT[vert].wNormal.x);
  }
    
  bool isVertexUsed(uint vert)
  {
    return OUT[vert].wNormal.y != 0;
  }

  void clearVertexUsed(uint vert) {
    OUT[vert].wNormal.y = 0;
  }

  void setVertexUsed(uint vert) {
    OUT[vert].wNormal.y = 1;
  }
  
  uint getVertexClip(uint vert) {
    return floatBitsToUint(OUT[vert].wNormal.z);
  }

  void setVertexClip(uint vert, uint mask) {
  #if USE_MESH_FRUSTUMCULL
    OUT[vert].wNormal.z = uintBitsToFloat(mask);
  #endif
  }
#else
  void clearVertexUsed(uint vert) {
    // dummy
  }
  void setVertexUsed(uint vert) {
    // dummy
  }
  void setVertexClip(uint vert, uint mask) {
    // dummy
  }
  uint getVertexClip(uint vert) {
    return getCullBits(gl_MeshVerticesNV[vert].gl_Position);
  }
#endif

  vec2 getVertexScreen(uint vert) {
    return getScreenPos(gl_MeshVerticesNV[vert].gl_Position);
  }

  #define NVMSH_BARRIER() \
    memoryBarrierShared(); \
    barrier();
    
  #define NVMSH_INDEX_BITS      8
  #define NVMSH_PACKED4X8_GET(packed, idx)   (((packed) >> (NVMSH_INDEX_BITS * (idx))) & 255)
  
  
  // only for tight packing case, 8 indices are loaded per thread
  #define NVMSH_PRIMITIVE_INDICES_RUNS  ((NVMESHLET_PRIMITIVE_COUNT * 3 + GROUP_SIZE * 8 - 1) / (GROUP_SIZE * 8))

  // processing loops
  #define NVMSH_VERTEX_RUNS     ((NVMESHLET_VERTEX_COUNT + GROUP_SIZE - 1) / GROUP_SIZE)
  #define NVMSH_PRIMITIVE_RUNS  ((NVMESHLET_PRIMITIVE_COUNT + GROUP_SIZE - 1) / GROUP_SIZE)
  
#if 1
  #define nvmsh_writePackedPrimitiveIndices4x8NV writePackedPrimitiveIndices4x8NV
#else
  #define nvmsh_writePackedPrimitiveIndices4x8NV(idx, topology) {\
        gl_PrimitiveIndicesNV[ (idx) + 0 ] = (NVMSH_PACKED4X8_GET((topology), 0)); \
        gl_PrimitiveIndicesNV[ (idx) + 1 ] = (NVMSH_PACKED4X8_GET((topology), 1)); \
        gl_PrimitiveIndicesNV[ (idx) + 2 ] = (NVMSH_PACKED4X8_GET((topology), 2)); \
        gl_PrimitiveIndicesNV[ (idx) + 3 ] = (NVMSH_PACKED4X8_GET((topology), 3));} 
#endif
  
///////////////////////////////////////////////////////////////////////////////

void main()
{
  // decode meshletDesc
  uvec4 desc = meshletDescs[meshletID + geometryOffsets.x];
  uint vertMax;
  uint primMax;
  uint vertBegin;
  uint primBegin;
  decodeMeshlet(desc, vertMax, primMax, vertBegin, primBegin);
  uint primCount = primMax + 1;
  uint vertCount = vertMax + 1;
  

  // LOAD PHASE
  
  // VERTEX PROCESSING
  for (uint i = 0; i < uint(NVMSH_VERTEX_RUNS); i++) {
    
    uint vert = laneID + i * GROUP_SIZE;
    
    clearVertexUsed( vert );
    
    // Use "min" to avoid branching
    // this ensures the compiler can batch loads
    // prior writes/processing
    //
    // Most of the time we will have fully saturated vertex utilization,
    // but we may compute the last vertex redundantly.
    {
      uint vidx = texelFetch(texIbo, int(vertBegin + min(vert,vertMax) + geometryOffsets.z)).x + geometryOffsets.w;
      vec4 hPos = procVertex(vert, vidx);
      setVertexClip(vert, getCullBits(hPos));
    
    #if USE_EARLY_ATTRIBUTES
      procAttributes(vert, vidx);
    #else
      writeVertexIndex(vert, vidx);
    #endif
    }
  }
  
  // PRIMITIVE TOPOLOGY
  // there are three different packing rules atm
  // FITTED_UINT8 gives fastest code and best bandwidth usage, at the sacrifice of
  // not maximizing actual primCount (primCount within meshlet may be always smaller than NVMESHLET_MAX_PRIMITIVES)
  // TIGHT_UINT8 is the next best thing with slighly more complex code to load
  // TRIANGLE_UINT32 provides the easist access, one primitive every 4 uint8 values, but wastes bandwidth
#if  NVMESHLET_PACKING_FITTED_UINT8
  // each run does read 8 indices per thread
  // the number of primCount was clamped in such fashion in advance
  // that it's guaranteed that gl_PrimiticeIndicesNV is sized big enough to allow the full 32-bit writes
  {
    uint readBegin = primBegin / 8 + geometryOffsets.y;
    uint readIndex = primCount * 3 - 1;
    uint readMax = readIndex / 8;

    for (uint i = 0; i < uint(NVMSH_PRIMITIVE_INDICES_RUNS); i++) {
      uint read = laneID + i * GROUP_SIZE;
      uint readUsed = min(read, readMax);
      //uvec2 topology = texelFetch(texPrim, int(readBegin + readUsed)).rg;
      uvec2 topology = primIndices[readBegin + readUsed];
      nvmsh_writePackedPrimitiveIndices4x8NV(readUsed * 8 + 0, topology.x);
      nvmsh_writePackedPrimitiveIndices4x8NV(readUsed * 8 + 4, topology.y);
    }
  }
#elif NVMESHLET_PACKING_TRIANGLE_UINT32
  {
    uint readBegin = primBegin / 4 + geometryOffsets.y;
    
    for (uint i = 0; i < uint(NVMSH_PRIMITIVE_RUNS); i++) {
      uint prim = laneID + i * GROUP_SIZE;
      
      prim = min(prim,primMax);
      {
        uint topology = texelFetch(texPrim, int(readBegin + prim)).x;
        
        uint idx = prim * 3;
        gl_PrimitiveIndicesNV[idx + 0] = NVMSH_PACKED4X8_GET(topology, 0);
        gl_PrimitiveIndicesNV[idx + 1] = NVMSH_PACKED4X8_GET(topology, 1);
        gl_PrimitiveIndicesNV[idx + 2] = NVMSH_PACKED4X8_GET(topology, 2);
      }
    }
  }
#else
  #error "NVMESHLET_PACKING not supported"
#endif

#if !USE_MESH_SHADERCULL
  if (laneID == 0) {
    gl_PrimitiveCountNV = primCount;
  #if USE_STATS
    atomicAdd(stats.meshletsOutput, 1);
    atomicAdd(stats.trisOutput, primCount);
    atomicAdd(stats.attrInput,  vertCount);
    atomicAdd(stats.attrOutput, vertCount);
  #endif
  }
#else
  uint outTriangles = 0;
  
  NVMSH_BARRIER();
  
  // PRIMITIVE PHASE
 
  const uint primRuns = (primCount + GROUP_SIZE - 1) / GROUP_SIZE;
  for (uint i = 0; i < primRuns; i++) {
    uint triCount = 0;
    uint topology = 0;
    
    uint prim = laneID + i * GROUP_SIZE;
    
    if (prim <= primMax) {
      uint idx = prim * 3;
      uint ia = gl_PrimitiveIndicesNV[idx + 0];
      uint ib = gl_PrimitiveIndicesNV[idx + 1];
      uint ic = gl_PrimitiveIndicesNV[idx + 2];
      topology = ia | (ib << NVMSH_INDEX_BITS) | (ic << (NVMSH_INDEX_BITS*2));
      
      // build triangle
      vec2 a = getVertexScreen(ia);
      vec2 b = getVertexScreen(ib);
      vec2 c = getVertexScreen(ic);

    #if USE_MESH_FRUSTUMCULL
      uint abits = getVertexClip(ia);
      uint bbits = getVertexClip(ib);
      uint cbits = getVertexClip(ic);
      
      triCount = testTriangle(a.xy, b.xy, c.xy, 1.0, abits, bbits, cbits);
    #else
      triCount = testTriangle(a.xy, b.xy, c.xy, 1.0, false);
    #endif
      
      if (triCount != 0) {
        setVertexUsed(ia);
        setVertexUsed(ib);
        setVertexUsed(ic);
      }
    }
    
  #if IS_VULKAN
    uvec4 vote = subgroupBallot(triCount == 1);
    uint  tris = subgroupBallotBitCount(vote);
    uint  idxOffset = outTriangles + subgroupBallotExclusiveBitCount(vote);
  #else
    uint vote = ballotThreadNV(triCount == 1);
    uint tris = bitCount(vote);
    uint idxOffset = outTriangles + bitCount(vote & gl_ThreadLtMaskNV);
  #endif
  
    if (triCount != 0) {
      uint idx = idxOffset * 3;
      gl_PrimitiveIndicesNV[idx + 0] = NVMSH_PACKED4X8_GET(topology, 0);
      gl_PrimitiveIndicesNV[idx + 1] = NVMSH_PACKED4X8_GET(topology, 1);
      gl_PrimitiveIndicesNV[idx + 2] = NVMSH_PACKED4X8_GET(topology, 2);
    }
    
    outTriangles += tris;
  }

  
  NVMSH_BARRIER();
  
  if (laneID == 0) {
    gl_PrimitiveCountNV = outTriangles;
  #if USE_STATS
    atomicAdd(stats.meshletsOutput, 1);
    atomicAdd(stats.trisOutput, outTriangles);
    #if USE_EARLY_ATTRIBUTES
      atomicAdd(stats.attrInput,  vertCount);
      atomicAdd(stats.attrOutput, vertCount);
    #endif
  #endif
  }

#if !USE_EARLY_ATTRIBUTES
  // FETCH REST OF VERTEX ATTRIBS
  
  #if USE_BATCHED_LATE_FETCH
  {
    // use two dedicated phases, which reduces the 
    // overall amount of latency, if compiler 
    // is smart enough to keep temp in registers
    // and not use local memory.
    //
    // - load  run R
    // - load  run R+1
    // ...
    // ! wait for loads
    // - write run R
    // - write run R+1
    // ...
    
    TempAttributes tempattrs[NVMSH_VERTEX_RUNS];
    
    for (uint i = 0; i < uint(NVMSH_VERTEX_RUNS); i++) {
      uint vert = laneID + i * GROUP_SIZE;
      bool used = isVertexUsed( vert );
      if (used) {
        uint vidx = readVertexIndex( vert );
        fetchAttributes(tempattrs[i], vert, vidx);
      }
    }
    for (uint i = 0; i < uint(NVMSH_VERTEX_RUNS); i++) {
      uint vert = laneID + i * GROUP_SIZE;
      bool used = isVertexUsed( vert );
      if (used) {
        uint vidx = readVertexIndex( vert );
        storeAttributes(tempattrs[i],vert, vidx);
      }
    }
  }
  #else
  {
    // due to dynamic branching the compiler may not unroll 
    // all loads prior all writes, which adds latency 
    // - load  run R
    // ! wait for loads
    // - write run R
    // - load  run R+1
    // ! wait for loads
    // - write run R+1
    // ...
    //
    // FIXME get compiler to do above with simple code
  
    for (uint i = 0; i < uint(NVMSH_VERTEX_RUNS); i++) {
      uint vert = laneID + i * GROUP_SIZE;
      bool used = isVertexUsed( vert );
      if (used) {
        uint vidx = readVertexIndex( vert );
        procAttributes(vert, vidx);
      }
    }
  }
  #endif
  
  #if USE_STATS
  {
    uint usedVertices = 0;
    for (uint i = 0; i < uint(NVMSH_VERTEX_RUNS); i++) {
      uint vert = laneID + i * GROUP_SIZE;
      bool used = isVertexUsed( vert );
    #if IS_VULKAN
      uvec4 vote  = subgroupBallot(used);
      uint  verts = subgroupBallotBitCount(vote);
    #else
      uint vote  = ballotThreadNV(used);
      uint verts = bitCount(vote);
    #endif
      usedVertices += verts;
    }
    if (laneID == 0){
      atomicAdd(stats.attrInput, vertCount);
      atomicAdd(stats.attrOutput, usedVertices);
    }
  }
  #endif
  
#endif // !USE_EARLY_ATTRIBUTES
#endif // !USE_MESH_SHADERCULL
}
