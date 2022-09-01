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

#version 460

  #extension GL_GOOGLE_include_directive : enable
  #extension GL_EXT_control_flow_attributes: require
  #define UNROLL_LOOP [[unroll]]

#include "config.h"

//////////////////////////////////////

  #extension GL_EXT_mesh_shader : require

//////////////////////////////////////

  #extension GL_EXT_shader_explicit_arithmetic_types_int8  : require
  #extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
  
  #extension GL_KHR_shader_subgroup_basic : require
  #extension GL_KHR_shader_subgroup_ballot : require
  #extension GL_KHR_shader_subgroup_vote : require

//////////////////////////////////////

#include "common.h"

//////////////////////////////////////////////////
// MESH CONFIG

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

// prefer load into shared memory, then work with data in separate pass
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
// defines how much information we store in shared memory
// for the vertices. We need them in shared memory so
// that primitive culling can access all vertices a
// primitive uses.
// One big difference to NV code is that EXT does not
// allow read-access to output data

// store screen position, use less shared memory and 
// speeds up primitive culling, but may need to 
// re-fetch/transform vertex position again.
#define HW_TEMPVERTEX_SPOS 0

// store world position and avoid the later re-fetch
// but during primitive culling need to transform all 3 vertices.
#define HW_TEMPVERTEX_WPOS 1


#if EXT_USE_ANY_COMPACTION
  // experiment with what store type is quicker
  #define HW_TEMPVERTEX  HW_TEMPVERTEX_SPOS
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
#elif HW_TEMPVERTEX == HW_TEMPVERTEX_SPOS
  vec3 wPos;
#endif
#if USE_VERTEX_CULL || EXT_USE_ANY_COMPACTION
  uint vidx;
#endif
};

// as this shader alywas does per-primitive culling
// we need to able to fetch the vertex screen positions
shared TempVertex s_tempVertices[NVMESHLET_VERTEX_COUNT];

#if EXT_USE_ANY_COMPACTION
// when we do any form of compaction, we will need a working
// set of primitives in shared memory, otherwise we can straight
// write to final outputs
shared u8vec4     s_tempPrimitives[NVMESHLET_PRIMITIVE_COUNT];
#endif

#if USE_VERTEX_CULL && EXT_COMPACT_VERTEX_OUTPUT
// for compacted vertices we also need to re-index the local
// triangle indices, from old vertex index to output vertex index
shared uint8_t    s_remapVertices[NVMESHLET_VERTEX_COUNT];
#endif

#if EXT_MESH_SUBGROUP_COUNT > 1
// if more than one subgroup is used, we need to sync total
// number of outputs via shared memory
shared uint s_outPrimCount;
shared uint s_outVertCount;
#endif

#if USE_VERTEX_CULL

  // we encode vertex usage in the highest bit of vidx
  // assuming it is available

  bool vertexcull_isVertexUsed(uint vert)
  {
    return (s_tempVertices[vert].vidx & (1<<31)) != 0;
  }

  void vertexcull_setVertexUsed(uint vert) {
    // non-atomic write as read/write hazard should not be
    // an issue here, this function will always just
    // add the topmost bit
    s_tempVertices[vert].vidx |= (1<<31);
  }
  
  uint vertexcull_readVertexIndex(uint vert) {
    return (s_tempVertices[vert].vidx & ((1<<31)-1));
  }
  
#elif EXT_USE_ANY_COMPACTION

  uint vertexcull_readVertexIndex(uint vert) {
    return s_tempVertices[vert].vidx;
  }

#endif
  
#if HW_TEMPVERTEX == HW_TEMPVERTEX_SPOS
  vec2 primcull_getVertexSPos(uint vert) {
    return s_tempVertices[vert].sPos;
  }
#elif HW_TEMPVERTEX == HW_TEMPVERTEX_WPOS
  vec4 primcull_getVertexHPos(uint vert) {
    return (scene.viewProjMatrix * vec4(s_tempVertices[vert].wPos,1));
  }
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
  s_tempVertices[vert].sPos = getScreenPos(hPos);
#elif HW_TEMPVERTEX == HW_TEMPVERTEX_WPOS
  s_tempVertices[vert].wPos = wPos;
#else
  #error "HW_TEMPVERTEX not supported"
#endif
#if USE_VERTEX_CULL || EXT_USE_ANY_COMPACTION
  s_tempVertices[vert].vidx = vidx;
#endif
}

#if EXT_USE_ANY_COMPACTION
void procTempVertex(uint vert, const uint vidx)
{
  vec3 oPos = getPosition(vidx);
  vec3 wPos = (object.worldMatrix  * vec4(oPos,1)).xyz;
  vec4 hPos = (scene.viewProjMatrix * vec4(wPos,1));
  
  // only early out if we could make out-of-bounds write
  if ((WORKGROUP_SIZE * MESHLET_VERTEX_ITERATIONS > NVMESHLET_VERTEX_COUNT) && vert >= NVMESHLET_VERTEX_COUNT) return;
  
  writeTempVertex(vert, vidx, wPos, hPos);
}
#endif

void procVertex(uint vert, const uint vidx)
{
#if HW_TEMPVERTEX == HW_TEMPVERTEX_SPOS || !EXT_USE_ANY_COMPACTION
  vec3 oPos = getPosition(vidx);
  vec3 wPos = (object.worldMatrix  * vec4(oPos,1)).xyz;
#elif HW_TEMPVERTEX == HW_TEMPVERTEX_WPOS
  vec3 wPos = s_tempVertices[vert].wPos;
#else
  #error "HW_TEMPVERTEX not supported"
#endif
  vec4 hPos = (scene.viewProjMatrix * vec4(wPos,1));
  
  uint inVert = vert;
  
#if USE_VERTEX_CULL && EXT_COMPACT_VERTEX_OUTPUT && !EXT_LOCAL_INVOCATION_VERTEX_OUTPUT
  // write to a different output location
  vert = s_remapVertices[vert];
#endif
  
  // only early out if we could make out-of-bounds write
  if ((WORKGROUP_SIZE * MESHLET_VERTEX_ITERATIONS > NVMESHLET_VERTEX_COUNT) && vert >= NVMESHLET_VERTEX_COUNT) return;

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
  
#if USE_VERTEX_CULL && EXT_COMPACT_VERTEX_OUTPUT && !EXT_LOCAL_INVOCATION_VERTEX_OUTPUT
  // write to a different output location
  vert = s_remapVertices[vert];
#endif
  
  // only early out if we could make out-of-bounds write
  if ((WORKGROUP_SIZE * MESHLET_VERTEX_ITERATIONS > NVMESHLET_VERTEX_COUNT) && vert >= NVMESHLET_VERTEX_COUNT) return;
  
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
#if EXT_MESH_SUBGROUP_COUNT > 1
  if (laneID == 0)
  {
    s_outVertCount  = 0;
    s_outPrimCount = 0;
  }
#endif

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
          procVertex(vert, vidx);
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
    
      if (prim <= primMax) {
        s_tempPrimitives[prim] = topology;
      }
    }
  #endif
  }

#else
  #error "NVMESHLET_ENCODING not supported"
#endif

  ////////////////////////////////////////////
  // PRIMITIVE CULLING PHASE
  
  memoryBarrierShared();
  barrier();
  
  // for pipelining the index loads it is actually faster to load
  // the primitive indices first, and then do the culling loop here,
  // rather than combining load / culling. This behavior, however, 
  // could vary per vendor.
  
  uint outPrimCount = 0;
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
    if (prim <= primMax) {
      uint idx = prim * 3;
      topology = s_tempPrimitives[prim];
    }
    #if EXT_MESH_SUBGROUP_COUNT > 1 && EXT_COMPACT_PRIMITIVE_OUTPUT
      // when we compact we will write topology to a new location in
      // s_tempPrimitives, so must ensure all threads have read the topology register properly
      barrier();
    #endif
  #else
    uint primRead = min(prim, primMax);
    topology = u8vec4(primIndices_u8[readBegin + primRead * 3 + 0],
                      primIndices_u8[readBegin + primRead * 3 + 1],
                      primIndices_u8[readBegin + primRead * 3 + 2],
                      uint8_t(prim));
  #endif

    if (prim <= primMax) {
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

    #if USE_MESH_FRUSTUMCULL && HW_TEMPVERTEX != HW_TEMPVERTEX_SPOS
      // if the task-shader is active and does the frustum culling
      // then we normally don't execute this here
      uint abits = getCullBits(ah);
      uint bbits = getCullBits(bh);
      uint cbits = getCullBits(ch);

      primVisible = testTriangle(as.xy, bs.xy, cs.xy, 1.0, abits, bbits, cbits);
    #else
      primVisible = testTriangle(as.xy, bs.xy, cs.xy, 1.0, false);
    #endif
      
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
    #elif !EXT_COMPACT_PRIMITIVE_OUTPUT
      if (!primVisible) {
        s_tempPrimitives[prim] = u8vec4(0);
      }
    #endif

    #if USE_VERTEX_CULL
      if (primVisible) {
        vertexcull_setVertexUsed(topology.x);
        vertexcull_setVertexUsed(topology.y);
        vertexcull_setVertexUsed(topology.z);
      }
    #endif
    }

  #if EXT_COMPACT_PRIMITIVE_OUTPUT || USE_STATS
    {
      uvec4 votePrims = subgroupBallot(primVisible);
      uint  numPrims  = subgroupBallotBitCount(votePrims);
      
    #if EXT_MESH_SUBGROUP_COUNT > 1
      if (gl_SubgroupInvocationID == 0) {
        outPrimCount = atomicAdd(s_outPrimCount, numPrims);
      }
      outPrimCount = subgroupBroadcastFirst(outPrimCount);
    #endif
      
    #if EXT_COMPACT_PRIMITIVE_OUTPUT
      uint  idxOffset = subgroupBallotExclusiveBitCount(votePrims) + outPrimCount;
      if (primVisible) {
        s_tempPrimitives[idxOffset] = topology;
      }
    #endif
    #if EXT_MESH_SUBGROUP_COUNT == 1
      outPrimCount += numPrims;
    #endif
    }
  #endif
  }
  
  
#if USE_VERTEX_CULL && (EXT_COMPACT_VERTEX_OUTPUT || USE_STATS)
  ////////////////////////////////////////////
  // VERTEX COMPACTION PHASE
  
  memoryBarrierShared();
  barrier();
  
  uint outVertCount = 0;
  {
    UNROLL_LOOP
    for (uint i = 0; i < uint(MESHLET_VERTEX_ITERATIONS); i++) {
      uint vert = laneID + i * WORKGROUP_SIZE;
      bool used = vert <= vertMax && vertexcull_isVertexUsed( vert );
      
    #if EXT_COMPACT_VERTEX_OUTPUT && EXT_LOCAL_INVOCATION_VERTEX_OUTPUT
      // ensure vtx is in register before we start
      // writing it into another shared memory location 
      // after compaction
      #if HW_TEMPVERTEX == HW_TEMPVERTEX_SPOS
        // only vidx matters
        uint vidx;
        if (used) vidx = s_tempVertices[vert].vidx;
      #else
        TempVertex vtx;
        if (used) vtx = s_tempVertices[vert];
      #endif
      #if EXT_MESH_SUBGROUP_COUNT > 1
        barrier();
      #endif
    #endif

      uvec4 voteVerts = subgroupBallot(used);
      uint  numVerts  = subgroupBallotBitCount(voteVerts);
      
    #if EXT_MESH_SUBGROUP_COUNT > 1
      if (gl_SubgroupInvocationID == 0) {
        outVertCount = atomicAdd(s_outVertCount, numVerts);
      }
      outVertCount = subgroupBroadcastFirst(outVertCount);
    #endif
      
    #if EXT_COMPACT_VERTEX_OUTPUT
      uint  idxOffset = subgroupBallotExclusiveBitCount(voteVerts) + outVertCount;   
      if (used) {
        #if EXT_LOCAL_INVOCATION_VERTEX_OUTPUT
          #if HW_TEMPVERTEX == HW_TEMPVERTEX_SPOS
            // only vidx matters
            s_tempVertices[idxOffset].vidx = vidx;
          #else
            s_tempVertices[idxOffset] = vtx;
          #endif
        #endif
        // we need to fix up the primitive indices
        // from old vertex index to compacted vertex index
        s_remapVertices[vert] = uint8_t(idxOffset);
      }
    #endif
      
    #if EXT_MESH_SUBGROUP_COUNT == 1
      outVertCount += numVerts;
    #endif
    }
  }
  
#else
  uint outVertCount = vertCount;
#endif

  ////////////////////////////////////////////
  // OUTPUT
  
  memoryBarrierShared();
  barrier();
  
  #if EXT_MESH_SUBGROUP_COUNT > 1
    outVertCount = s_outVertCount;
    outPrimCount = s_outPrimCount;
  #endif

  if (laneID == 0) {
  #if USE_STATS
    atomicAdd(stats.meshletsOutput, 1);
    atomicAdd(stats.trisOutput, outPrimCount);
    atomicAdd(stats.attrInput,  vertCount);
    atomicAdd(stats.attrOutput, outVertCount);
  #endif
  }
  
#if !EXT_COMPACT_PRIMITIVE_OUTPUT
  outPrimCount = primCount;
#endif
#if !EXT_COMPACT_VERTEX_OUTPUT
  outVertCount = vertCount;
#endif

#if EXT_USE_ANY_COMPACTION
  // OUTPUT ALLOCATION
  SetMeshOutputsEXT(outVertCount, outPrimCount);
  
  // OUTPUT TRIANGLES
  UNROLL_LOOP
  for (uint i = 0; i < uint(MESHLET_PRIMITIVE_ITERATIONS); i++)
  {
    uint prim = laneID + i * WORKGROUP_SIZE;
    if (prim < outPrimCount) {
      u8vec4 topology = s_tempPrimitives[prim];
    #if USE_VERTEX_CULL && EXT_COMPACT_VERTEX_OUTPUT
      // re-index vertices to new output vertex slots
      topology.x = s_remapVertices[topology.x];
      topology.y = s_remapVertices[topology.y];
      topology.z = s_remapVertices[topology.z];
    #endif
      gl_PrimitiveTriangleIndicesEXT[prim] = uvec3(topology.x, topology.y, topology.z);
    #if SHOW_PRIMIDS
      // let's compute some fake unique primitiveID
      gl_MeshPrimitivesEXT[prim].gl_PrimitiveID = int((meshletID + geometryOffsets.x) * NVMESHLET_PRIMITIVE_COUNT + uint(topology.w));
    #endif
    }
  }
#endif

#if USE_VERTEX_CULL || EXT_USE_ANY_COMPACTION
  // OUTPUT VERTICES
  UNROLL_LOOP
  for (uint i = 0; i < uint(MESHLET_VERTEX_ITERATIONS); i++)
  {
    uint vert = laneID + i * WORKGROUP_SIZE;
  #if USE_VERTEX_CULL && EXT_COMPACT_VERTEX_OUTPUT && EXT_LOCAL_INVOCATION_VERTEX_OUTPUT
    bool used = vert < outVertCount;
  #elif USE_VERTEX_CULL
    bool used = vert <= vertMax && vertexcull_isVertexUsed( vert );
  #else
    bool used = vert <= vertMax;
  #endif
    if (used) {
      uint vidx = vertexcull_readVertexIndex( vert );
    #if EXT_USE_ANY_COMPACTION
      procVertex(vert, vidx);
    #endif
      procAttributes(vert, vidx);
    }
  }
#endif
}
