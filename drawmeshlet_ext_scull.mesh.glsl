/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */


// This shader is currently not used stand-alone
// but used conditionally be `drawmeshlet_ext_cull.mesh.glsl`.
// Though it can be made stand-alone easily.
//
// It is an optimized version of `drawmeshlet_ext_cull.mesh.glsl`
// for the vertex and/or triangle counts being 32 or 64 as well
// as the subgroup size being 32 or 64.
//
// It does not use shared memory, but uses shuffle instead to
// handle re-ordering of data. Such variables will be stored
// in "temp" variables as registers.


#include "config.h"

//////////////////////////////////////

  #extension GL_EXT_mesh_shader : require

//////////////////////////////////////

  #extension GL_EXT_shader_explicit_arithmetic_types_int8  : require
  #extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
  #extension GL_EXT_shader_explicit_arithmetic_types_int32 : require
  #extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
  #extension GL_EXT_shader_subgroup_extended_types_int64 : require
  
  #extension GL_KHR_shader_subgroup_basic : require
  #extension GL_KHR_shader_subgroup_ballot : require
  #extension GL_KHR_shader_subgroup_vote : require
  #extension GL_KHR_shader_subgroup_shuffle : require
  #extension GL_KHR_shader_subgroup_arithmetic : require

//////////////////////////////////////

#include "common.h"

//////////////////////////////////////////////////
// MESH CONFIG

#if EXT_MESH_SUBGROUP_COUNT != 1
#error "EXT_MESH_SUBGROUP_COUNT must be 1 in this shader"
#endif
#if !(NVMESHLET_VERTEX_COUNT == 64 ||  NVMESHLET_VERTEX_COUNT == 32)
#error "NVMESHLET_VERTEX_COUNT must be 32 or 64 in this shader"
#endif
#if !(NVMESHLET_PRIMITIVE_COUNT == 64 ||  NVMESHLET_PRIMITIVE_COUNT == 32)
#error "NVMESHLET_PRIMITIVE_COUNT must be 32 or 64 in this shader"
#endif


// see Sample::getShaderPrepend() how these are computed
const uint WORKGROUP_SIZE = EXT_MESH_SUBGROUP_COUNT * EXT_MESH_SUBGROUP_SIZE;

layout(local_size_x=WORKGROUP_SIZE) in;
layout(max_vertices=NVMESHLET_VERTEX_COUNT, max_primitives=NVMESHLET_PRIMITIVE_COUNT) out;
layout(triangles) out;

// The workgroup size of the shader may not have enough threads
// to do all the work in a unique thread.
// Therefore we might need to loop to process all the work.
// Looping can have the benefit that we can amortize some registers
// that are common to all threads. However, it may also introduce
// more registers. 

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
#define USE_VERTEX_CULL    1
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

#if (SHOW_PRIMIDS || USE_BARYCENTRIC_SHADING) && (!EXT_COMPACT_VERTEX_OUTPUT)
// no attributes exist in these modes, so disable vertex culling, unless compact is preferred
#undef  USE_VERTEX_CULL
#define USE_VERTEX_CULL  0
#endif

// mostly used to detect if no compactiong is active at all, then we will pre-allocate similar to basic shader
#define EXT_USE_ANY_COMPACTION  ((USE_VERTEX_CULL && EXT_COMPACT_VERTEX_OUTPUT) || EXT_COMPACT_PRIMITIVE_OUTPUT)

// prefer load into registers, then work with data in separate pass
// should make sense if a single subgroup has to loop
#ifndef USE_EARLY_TOPOLOGY_LOAD
#define USE_EARLY_TOPOLOGY_LOAD  ((EXT_MESH_SUBGROUP_COUNT == 1) && (NVMESHLET_PRIMITIVE_COUNT > EXT_MESH_SUBGROUP_SIZE))
#endif


/////////////////////////////////////
// UNIFORMS

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
    uint8_t primIndices_u8[];
  };

  layout(binding=GEOMETRY_TEX_VBO,  set=DSET_GEOMETRY)  uniform samplerBuffer  texVbo;
  layout(binding=GEOMETRY_TEX_ABO,  set=DSET_GEOMETRY)  uniform samplerBuffer  texAbo;

/////////////////////////////////////////////////

#include "nvmeshlet_utils.glsl"

uint findNthBit(uint value, uint n)
{
  // from https://stackoverflow.com/a/45487375

  const uint  pop2 = (value & 0x55555555u) + ((value >> 1) & 0x55555555u);
  const uint  pop4 = (pop2 & 0x33333333u) + ((pop2 >> 2) & 0x33333333u);
  const uint  pop8 = (pop4 & 0x0f0f0f0fu) + ((pop4 >> 4) & 0x0f0f0f0fu);
  const uint  pop16 = (pop8 & 0x00ff00ffu) + ((pop8 >> 8) & 0x00ff00ffu);
  const uint  pop32 = (pop16 & 0x000000ffu) + ((pop16 >> 16) & 0x000000ffu);
  uint        rank = 0;
  uint        temp;

  if (n++ >= pop32)
    return 31; // avoid out of bounds, so report 31 even when not found

  temp = pop16 & 0xffu;
  /* if (n > temp) { n -= temp; rank += 16; } */
  rank += ((temp - n) & 256) >> 4;
  n -= temp & ((temp - n) >> 8);

  temp = (pop8 >> rank) & 0xffu;
  /* if (n > temp) { n -= temp; rank += 8; } */
  rank += ((temp - n) & 256) >> 5;
  n -= temp & ((temp - n) >> 8);

  temp = (pop4 >> rank) & 0x0fu;
  /* if (n > temp) { n -= temp; rank += 4; } */
  rank += ((temp - n) & 256) >> 6;
  n -= temp & ((temp - n) >> 8);

  temp = (pop2 >> rank) & 0x03u;
  /* if (n > temp) { n -= temp; rank += 2; } */
  rank += ((temp - n) & 256) >> 7;
  n -= temp & ((temp - n) >> 8);

  temp = (value >> rank) & 0x01u;
  /* if (n > temp) rank += 1; */
  rank += ((temp - n) & 256) >> 8;

  return rank;
}

uint findNthBit(uint64_t v, uint i)
{
  uvec2 v2 = unpackUint2x32(v);
  uint s = v2.x;
  uint o = 0;
  uint bc = bitCount(v2.x);
  if (i >= bc) {
    i -= bc;
    o += 32;
    s = v2.y;
  }
  
  return findNthBit(s, i) + o;
}

uint myBitCount(uint32_t v)
{
  return bitCount(v);
}

uint myBitCount(uint64_t v)
{
  uvec2 v2 = unpackUint2x32(v);
  return bitCount(v2.x) + bitCount(v2.y);
}

/////////////////////////////////////////////////
// MESH INPUT

#if USE_TASK_STAGE
  struct Task {
    uint    baseID;
    uint8_t deltaIDs[NVMESHLET_PER_TASK];
  };  
  taskPayloadSharedEXT Task IN;
  
  // gl_WorkGroupID.x runs from [0 .. parentTask.groupCountX - 1]
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

// some of this HW defines may be vendor specific preference at the moment

// use gl_CullPrimitiveEXT if applicable otherwise degenerate
// FIXME NVIDIA need to optimize 1 case
#ifndef HW_CULL_PRIMITIVE
#define HW_CULL_PRIMITIVE 1
#endif

// HW_TEMPVERTEX
// defines how much information we store in temp registers
// for the vertices. We need them in temp registers so
// that primitive culling can access all vertices a
// primitive uses via shuffle.
// One big difference to NV code is that EXT does not
// allow read-access to output data

// store screen position, use less temp registers and 
// speeds up primitive culling, but may need to 
// re-fetch/transform vertex position again.
#define HW_TEMPVERTEX_SPOS 0

// store world position and avoid the later re-fetch
// but during primitive culling need to transform all 3 vertices.
#define HW_TEMPVERTEX_WPOS 1


#if EXT_USE_ANY_COMPACTION
  // experiment with what store type is quicker
  #define HW_TEMPVERTEX  HW_TEMPVERTEX_WPOS
#else
  // without compaction
  // always use smallest here, as vertex wpos
  // is not used later at all
  #define HW_TEMPVERTEX  HW_TEMPVERTEX_SPOS
#endif

struct TempVertex
{
#if HW_TEMPVERTEX == HW_TEMPVERTEX_SPOS
  vec2 sPos;
#elif HW_TEMPVERTEX == HW_TEMPVERTEX_WPOS
  vec3 wPos;
#endif
#if USE_VERTEX_CULL || EXT_USE_ANY_COMPACTION
  uint vidx;
#endif
};

// as this shader alywas does per-primitive culling
// we need to able to fetch the vertex screen positions
TempVertex tempVertices[MESHLET_VERTEX_ITERATIONS];

#if USE_VERTEX_CULL

#if NVMESHLET_VERTEX_COUNT == 64
  #define vertexBits_t uint64_t
#else
  #define vertexBits_t uint
#endif

  vertexBits_t tempVertexUsed = 0;

  // we encode vertex usage in the highest bit of vidx
  // assuming it is available

  bool vertexcull_isVertexUsed(uint vert)
  {
    return (tempVertexUsed & (vertexBits_t(1) << vert)) != 0;
  }
  
  uint vertexcull_postCompactIndex(uint vert)
  {
    return uint(myBitCount(tempVertexUsed & ((vertexBits_t(1) << vert)-1)));
  }
  
  uint vertexcull_preCompactIndex(uint overt)
  {
    return findNthBit(tempVertexUsed, overt);
  }

  void vertexcull_setVertexUsed(uint a) {
    // non-atomic write as read/write hazard should not be
    // an issue here, this function will always just
    // add the topmost bit
    tempVertexUsed |= vertexBits_t(1) << a;
  }
  
#endif
  
#if USE_VERTEX_CULL || EXT_USE_ANY_COMPACTION
  uint vertexcull_readVertexIndex(uint vert) {
    #if USE_VERTEX_CULL && EXT_COMPACT_VERTEX_OUTPUT && EXT_LOCAL_INVOCATION_VERTEX_OUTPUT
      #if EXT_MESH_SUBGROUP_SIZE == NVMESHLET_VERTEX_COUNT
        return subgroupShuffle(tempVertices[0].vidx, vert);
      #elif EXT_MESH_SUBGROUP_SIZE == 32
        uint lo = subgroupShuffle(tempVertices[0].vidx, vert & 31);
        uint hi = subgroupShuffle(tempVertices[1].vidx, vert & 31);
      
        return vert < 32 ? lo : hi;
      #endif
    #else
      #if EXT_MESH_SUBGROUP_SIZE == NVMESHLET_VERTEX_COUNT
        return tempVertices[0].vidx;
      #else
        return vert < 32 ? tempVertices[0].vidx : tempVertices[1].vidx;
      #endif
    #endif
  }
  
#endif

  uint tempTopologies[MESHLET_PRIMITIVE_ITERATIONS];

#if NVMESHLET_PRIMITIVE_COUNT == 64
  #define primBits_t uint64_t
#else
  #define primBits_t uint
#endif

  primBits_t tempPrimUsed = 0;

  uint primcull_preCompactIndex(uint overt)
  {
    return findNthBit(tempPrimUsed, overt);
  }
  
  uint primcull_getTopology(uint idx)
  {
    #if EXT_COMPACT_PRIMITIVE_OUTPUT && EXT_LOCAL_INVOCATION_PRIMITIVE_OUTPUT
      #if EXT_MESH_SUBGROUP_SIZE == NVMESHLET_PRIMITIVE_COUNT
        return subgroupShuffle(tempTopologies[0], idx);
      #else
        uint lo = subgroupShuffle(tempTopologies[0], idx & 31);
        uint hi = subgroupShuffle(tempTopologies[1], idx & 31);
      
        return idx < 32 ? lo : hi;
      #endif
    #else
      #if EXT_MESH_SUBGROUP_SIZE == NVMESHLET_PRIMITIVE_COUNT
        return tempTopologies[0];
      #else
        return idx < 32 ? tempTopologies[0] : tempTopologies[1];
      #endif
    #endif
  }

  
#if HW_TEMPVERTEX == HW_TEMPVERTEX_SPOS
  vec2 primcull_getVertexSPos(uint vert) {
    #if EXT_MESH_SUBGROUP_SIZE == NVMESHLET_VERTEX_COUNT
      return subgroupShuffle(tempVertices[0].sPos, vert);
    #else
      vec2 lo = subgroupShuffle(tempVertices[0].sPos, vert & 31);
      vec2 hi = subgroupShuffle(tempVertices[1].sPos, vert & 31);
    
      return vert < 32 ? lo : hi;
    #endif
  }
#elif HW_TEMPVERTEX == HW_TEMPVERTEX_WPOS
  vec4 primcull_getVertexHPos(uint vert) {
    #if EXT_MESH_SUBGROUP_SIZE == NVMESHLET_VERTEX_COUNT
      vec3 wPos = subgroupShuffle(tempVertices[0].wPos, vert);
    #else
      vec3 lo = subgroupShuffle(tempVertices[0].wPos, vert & 31);
      vec3 hi = subgroupShuffle(tempVertices[1].wPos, vert & 31);
    
      vec3 wPos = vert < 32 ? lo : hi;
    #endif
    
    return (scene.viewProjMatrix * vec4(wPos,1));
  }
  #if EXT_USE_ANY_COMPACTION
  vec3 primcull_getVertexWPos(uint vert) {
    #if USE_VERTEX_CULL && EXT_COMPACT_VERTEX_OUTPUT && EXT_LOCAL_INVOCATION_VERTEX_OUTPUT
      #if EXT_MESH_SUBGROUP_SIZE == NVMESHLET_VERTEX_COUNT
        vec3 wPos = subgroupShuffle(tempVertices[0].wPos, vert);
      #else
        vec3 lo = subgroupShuffle(tempVertices[0].wPos, vert & 31);
        vec3 hi = subgroupShuffle(tempVertices[1].wPos, vert & 31);
      
        vec3 wPos = vert < 32 ? lo : hi;
      #endif
    #else
      #if EXT_MESH_SUBGROUP_SIZE == NVMESHLET_VERTEX_COUNT
        vec3 wPos = tempVertices[0].wPos;
      #else
        vec3 wPos = vert < 32 ? tempVertices[0].wPos : tempVertices[1].wPos;
      #endif
    #endif
     
    return wPos;
  }
  #endif
#else
  #error "HW_TEMPVERTEX not supported"
#endif


//////////////////////////////////////////////////
// VERTEX EXECUTION

// This is the code that is normally done in the vertex-shader
// "vidx" is what gl_VertexIndex would be
//
// We split vertex-shading from attribute-shading,
// to highlight the differences between the drawmeshlet_cull.mesh.glsl
// and drawmeshlet_basic.mesh.glsl files (just use a file-diff
// program to view the two)

// the temp vertex is required for per-primitive culling
// as well as vertex culling

void writeTempVertex(uint vert, const uint vidx, vec3 wPos, vec4 hPos)
{
#if HW_TEMPVERTEX == HW_TEMPVERTEX_SPOS
  vec2 sPos = getScreenPos(hPos);
#endif

#if EXT_MESH_SUBGROUP_SIZE != NVMESHLET_VERTEX_COUNT
  if (vert < 32) {
#endif
#if HW_TEMPVERTEX == HW_TEMPVERTEX_SPOS
  tempVertices[0].sPos = sPos;
#elif HW_TEMPVERTEX == HW_TEMPVERTEX_WPOS
  tempVertices[0].wPos = wPos;
#else
  #error "HW_TEMPVERTEX not supported"
#endif
#if USE_VERTEX_CULL || EXT_USE_ANY_COMPACTION
  tempVertices[0].vidx = vidx;
#endif
#if EXT_MESH_SUBGROUP_SIZE != NVMESHLET_VERTEX_COUNT
  }
  else {
#if HW_TEMPVERTEX == HW_TEMPVERTEX_SPOS
  tempVertices[1].sPos = sPos;
#elif HW_TEMPVERTEX == HW_TEMPVERTEX_WPOS
  tempVertices[1].wPos = wPos;
#else
  #error "HW_TEMPVERTEX not supported"
#endif
#if USE_VERTEX_CULL || EXT_USE_ANY_COMPACTION
  tempVertices[1].vidx = vidx;
#endif
  }
#endif
}

#if EXT_USE_ANY_COMPACTION
void procTempVertex(uint vert, const uint vidx)
{
  vec3 oPos = getPosition(vidx);
  vec3 wPos = (object.worldMatrix  * vec4(oPos,1)).xyz;
  vec4 hPos = (scene.viewProjMatrix * vec4(wPos,1));
  
  writeTempVertex(vert, vidx, wPos, hPos);
}
#endif

void procVertex(uint vert, const uint vidx, vec3 inWPos)
{
#if HW_TEMPVERTEX == HW_TEMPVERTEX_SPOS || !EXT_USE_ANY_COMPACTION
  vec3 oPos = getPosition(vidx);
  vec3 wPos = (object.worldMatrix  * vec4(oPos,1)).xyz;
#elif HW_TEMPVERTEX == HW_TEMPVERTEX_WPOS
  vec3 wPos = inWPos;
#else
  #error "HW_TEMPVERTEX not supported"
#endif
  vec4 hPos = (scene.viewProjMatrix * vec4(wPos,1));
  
  gl_MeshVerticesEXT[vert].gl_Position = hPos;

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
  gl_MeshVerticesEXT[vert].gl_ClipDistance[0] = dot(scene.wClipPlanes[0], vec4(wPos,1));
  #endif
  #if NUM_CLIPPING_PLANES > 1
  gl_MeshVerticesEXT[vert].gl_ClipDistance[1] = dot(scene.wClipPlanes[1], vec4(wPos,1));
  #endif
  #if NUM_CLIPPING_PLANES > 2
  gl_MeshVerticesEXT[vert].gl_ClipDistance[2] = dot(scene.wClipPlanes[2], vec4(wPos,1));
  #endif
#else
  for (int i = 0; i < NUM_CLIPPING_PLANES; i++){
    gl_MeshVerticesEXT[vert].gl_ClipDistance[i] = dot(scene.wClipPlanes[i], vec4(wPos,1));
  }
#endif
#endif

#if !EXT_USE_ANY_COMPACTION
  // without any compaction we still need to write temp vertex
  // as it's used for primitive culling
  writeTempVertex(vert, vidx, wPos, hPos);
#endif
}


void procAttributes(uint vert, const uint vidx)
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
  

#if !EXT_USE_ANY_COMPACTION
  // OUTPUT ALLOCATION
  // no compaction whatsoever, pre-allocate space early, directly
  // fill in crucial data
  SetMeshOutputsEXT(vertCount, primCount);
#endif

  // VERTEX INITIAL PROCESSING
  {
    UNROLL_LOOP
    for (uint i = 0; i < uint(MESHLET_VERTEX_ITERATIONS); i++)
    {
      uint vert = laneID + i * WORKGROUP_SIZE;
      uint vertLoad = min(vert, vertMax);

      {
        uint idx   = (vertLoad) >> (vidxDiv-1);
        uint shift = (vertLoad) & (vidxDiv-1);

        uint vidx = primIndices1[idx + vidxStart];
        vidx <<= vidxBits * (1-shift);
        vidx >>= vidxBits;

        vidx += geometryOffsets.w;
        
        {
      #if EXT_USE_ANY_COMPACTION
          // for compaction we will make final writes
          // of vertices after primitive culling and at
          // this point only compute enough for primitive
          // culling
          procTempVertex(vert, vidx);
      #else
          // process our vertex in full, as we need it for
          // culling anyway
          procVertex(vert, vidx, vec3(0,0,0));
        #if !USE_VERTEX_CULL
          // without vertex culling just write
          // out all attributes immediately
          // otherwise will defer attributes.
          procAttributes(vert, vidx);
        #endif
      #endif
        }
      }
    }
  }
  

  // PRIMITIVE TOPOLOGY
  {
  #if (EXT_USE_ANY_COMPACTION && USE_EARLY_TOPOLOGY_LOAD)
    // with compaction we do all loads up-front
    
    uint readBegin = primStart * 4;
  
    UNROLL_LOOP
    for (uint i = 0; i < uint(MESHLET_PRIMITIVE_ITERATIONS); i++)
    {
      uint prim     = laneID + i * WORKGROUP_SIZE;
      uint primRead = min(prim, primMax);
      
      u8vec4 topology = u8vec4(primIndices_u8[readBegin + primRead * 3 + 0],
                               primIndices_u8[readBegin + primRead * 3 + 1],
                               primIndices_u8[readBegin + primRead * 3 + 2],
                               uint8_t(prim));
    
      tempTopologies[i] = pack32(topology);
    }
  #endif
  }

#else
  #error "NVMESHLET_ENCODING not supported"
#endif

  ////////////////////////////////////////////
  // PRIMITIVE CULLING PHASE
  
  barrier();
  
  // for pipelining the index loads it is actually faster to load
  // the primitive indices first, and then do the culling loop here,
  // rather than combining load / culling. This behavior, however, 
  // could vary per vendor.

#if !(EXT_USE_ANY_COMPACTION && USE_EARLY_TOPOLOGY_LOAD)
  uint readBegin = primStart * 4;

#endif
  
  UNROLL_LOOP
  for (uint i = 0; i < uint(MESHLET_PRIMITIVE_ITERATIONS); i++)
  {
    uint prim     = laneID + i * WORKGROUP_SIZE;

    
    bool   primVisible = false;
    u8vec4 topology;
    
  #if (EXT_USE_ANY_COMPACTION && USE_EARLY_TOPOLOGY_LOAD)
    topology = unpack8(tempTopologies[i]);
  #else
    uint primRead = min(prim, primMax);
    topology = u8vec4(primIndices_u8[readBegin + primRead * 3 + 0],
                      primIndices_u8[readBegin + primRead * 3 + 1],
                      primIndices_u8[readBegin + primRead * 3 + 2],
                      uint8_t(prim));
  #endif
    if (prim > primMax) {
      topology = u8vec4(0);
    }
    
    // these read via shuffle, cannot be in branch
    
    #if HW_TEMPVERTEX == HW_TEMPVERTEX_SPOS
      vec2 as = primcull_getVertexSPos(topology.x);
      vec2 bs = primcull_getVertexSPos(topology.y);
      vec2 cs = primcull_getVertexSPos(topology.z);
    #else
      // build triangle
      vec4 ah = primcull_getVertexHPos(topology.x);
      vec4 bh = primcull_getVertexHPos(topology.y);
      vec4 ch = primcull_getVertexHPos(topology.z);
      
      vec2 as = getScreenPos(ah);
      vec2 bs = getScreenPos(bh);
      vec2 cs = getScreenPos(ch);
    #endif
    
    if (prim <= primMax) {
    #if USE_MESH_FRUSTUMCULL && HW_TEMPVERTEX != HW_TEMPVERTEX_SPOS && USE_CULLBITS
      // if the task-shader is active and does the frustum culling
      // then we normally don't execute this here
      uint abits = getCullBits(ah);
      uint bbits = getCullBits(bh);
      uint cbits = getCullBits(ch);

      primVisible = testTriangle(as.xy, bs.xy, cs.xy, 1.0, abits, bbits, cbits);
    #elif USE_MESH_FRUSTUMCULL
    
      // the simple viewport culling here only does 2D check
      primVisible = testTriangle(as.xy, bs.xy, cs.xy, 1.0, true);
    #else
      
      // assumes all heavy lifting on frustum culling is done before
      // either by task-shader or indirect draws etc. (not used in this sample)
      primVisible = testTriangle(as.xy, bs.xy, cs.xy, 1.0, false);
    #endif
    

      if (primVisible) {
      #if USE_VERTEX_CULL
        vertexcull_setVertexUsed(topology.x);
        vertexcull_setVertexUsed(topology.y);
        vertexcull_setVertexUsed(topology.z);
      #endif
      }
    
      
    #if !EXT_USE_ANY_COMPACTION
      {
      #if HW_CULL_PRIMITIVE
        // use gl_CullPrimitiveEXT, write this prior other per-primitive outputs of same primitive
        gl_MeshPrimitivesEXT[prim].gl_CullPrimitiveEXT = !primVisible;
        gl_PrimitiveTriangleIndicesEXT[prim] = uvec3(topology.x, topology.y, topology.z); // avoid branch always write
      #else
        // use degenerate triangle indices
        gl_PrimitiveTriangleIndicesEXT[prim] = primVisible ? uvec3(topology.x, topology.y, topology.z) : uvec3(0);
      #endif
      #if SHOW_PRIMIDS
        // let's compute some fake unique primitiveID
        gl_MeshPrimitivesEXT[prim].gl_PrimitiveID = int((meshletID + geometryOffsets.x) * NVMESHLET_PRIMITIVE_COUNT + uint(topology.w));
      #endif
      }
    #elif (!USE_EARLY_TOPOLOGY_LOAD)
      tempTopologies[i] = pack32(topology);
    #endif
    }
    

  #if EXT_COMPACT_PRIMITIVE_OUTPUT || USE_VERTEX_CULL || USE_STATS
    {
      uvec4 votePrims = subgroupBallot(primVisible);
    #if NVMESHLET_PRIMITIVE_COUNT == 64 && EXT_MESH_SUBGROUP_SIZE == 64
      tempPrimUsed = packUint2x32(votePrims.xy); 
    #elif NVMESHLET_PRIMITIVE_COUNT == 64 && EXT_MESH_SUBGROUP_SIZE == 32
      tempPrimUsed |= primBits_t(votePrims.x) << (i*32);
    #else
      tempPrimUsed = votePrims.x; 
    #endif
    }
  #endif
  }
  
  tempPrimUsed = subgroupOr(tempPrimUsed);
  uint outPrimCount = myBitCount(tempPrimUsed);

  
#if USE_VERTEX_CULL
  ////////////////////////////////////////////
  // VERTEX COMPACTION PHASE
  
  tempVertexUsed = subgroupOr(tempVertexUsed);
  uint outVertCount = myBitCount(tempVertexUsed);
  
#else
  uint outVertCount = vertCount;
#endif

  ////////////////////////////////////////////
  // OUTPUT
  
  barrier();

  if (laneID == 0) {
  #if USE_STATS
    atomicAdd(stats.meshletsOutput, 1);
    atomicAdd(stats.trisOutput, outPrimCount);
    atomicAdd(stats.attrInput,  vertCount);
    atomicAdd(stats.attrOutput, outVertCount);
  #endif
  }
  
#if EXT_USE_ANY_COMPACTION

#if !EXT_COMPACT_PRIMITIVE_OUTPUT
  outPrimCount = primCount;
#endif
#if !EXT_COMPACT_VERTEX_OUTPUT
  outVertCount = vertCount;
#endif

  SetMeshOutputsEXT(outVertCount, outPrimCount);

  UNROLL_LOOP
  for (uint i = 0; i < uint(MESHLET_PRIMITIVE_ITERATIONS); i++)
  {
  #if EXT_COMPACT_PRIMITIVE_OUTPUT && !EXT_LOCAL_INVOCATION_PRIMITIVE_OUTPUT
    uint prim  = laneID + i * WORKGROUP_SIZE;
    u8vec4 topology = unpack8(tempTopologies[i]);
    primBits_t primBit = primBits_t(1) << (prim);
    if ((tempPrimUsed & primBit) != 0)
    {
      uint outidx = myBitCount(tempPrimUsed & (primBit-1));
    
  #elif EXT_COMPACT_PRIMITIVE_OUTPUT && EXT_LOCAL_INVOCATION_PRIMITIVE_OUTPUT
    uint outidx = laneID + i * WORKGROUP_SIZE;
    
    // must be outside branch when reading via shuffle
    uint prim = primcull_preCompactIndex(outidx);
    u8vec4 topology = unpack8(primcull_getTopology(prim));
    
    if (outidx < outPrimCount)
    {
      
  #else
    uint prim  = laneID + i * WORKGROUP_SIZE;
    u8vec4 topology = unpack8(tempTopologies[i]);
    primBits_t primBit = primBits_t(1) << (prim);
    if ((tempPrimUsed & primBit) != 0)
    {
      uint outidx = prim;
  #endif
    #if EXT_COMPACT_VERTEX_OUTPUT && USE_VERTEX_CULL
      uvec3 outTopo = uvec3(vertexcull_postCompactIndex(topology.x), vertexcull_postCompactIndex(topology.y), vertexcull_postCompactIndex(topology.z));
    #else
      uvec3 outTopo = uvec3(topology.x, topology.y, topology.z);
    #endif
    
      gl_PrimitiveTriangleIndicesEXT[outidx] = outTopo;
    #if SHOW_PRIMIDS
      // let's compute some fake unique primitiveID
      gl_MeshPrimitivesEXT[outidx].gl_PrimitiveID = int((meshletID + geometryOffsets.x) * NVMESHLET_PRIMITIVE_COUNT + uint(topology.w));
    #endif
    }  
  }
#endif

#if USE_VERTEX_CULL || EXT_USE_ANY_COMPACTION
  // OUTPUT VERTICES
  UNROLL_LOOP
  for (uint i = 0; i < uint(MESHLET_VERTEX_ITERATIONS); i++)
  {
  #if USE_VERTEX_CULL && EXT_COMPACT_VERTEX_OUTPUT && !EXT_LOCAL_INVOCATION_VERTEX_OUTPUT
    uint vert = laneID + i * WORKGROUP_SIZE;
    bool used = vert <= vertMax && vertexcull_isVertexUsed( vert );
    if (used)
    {
      uint overt = vertexcull_postCompactIndex(vert);
      uint vidx  = vertexcull_readVertexIndex(vert);
      vec3 wPos  = primcull_getVertexWPos(vert);
      
  #elif USE_VERTEX_CULL && EXT_COMPACT_VERTEX_OUTPUT && EXT_LOCAL_INVOCATION_VERTEX_OUTPUT
    uint overt = laneID + i * WORKGROUP_SIZE;
    
    // must be outside branch when reading via shuffle
    uint vert  = vertexcull_preCompactIndex(overt);
    uint vidx  = vertexcull_readVertexIndex(vert);
    vec3 wPos  = primcull_getVertexWPos(vert);
    
    if (overt < outVertCount)
    {
    
  #else
    uint vert = laneID + i * WORKGROUP_SIZE;
    #if USE_VERTEX_CULL
    bool used = vert <= vertMax && vertexcull_isVertexUsed( vert );
    #else
    bool used = vert <= vertMax;
    #endif
    if (used)
    {
      uint overt = vert;
      uint vidx  = vertexcull_readVertexIndex( vert );
    #if EXT_USE_ANY_COMPACTION
      vec3 wPos  = primcull_getVertexWPos(vert);
    #endif
  #endif
    
    #if EXT_USE_ANY_COMPACTION
      procVertex(overt, vidx, wPos);
    #endif
      procAttributes(overt, vidx);
    }
  }
#endif
}
