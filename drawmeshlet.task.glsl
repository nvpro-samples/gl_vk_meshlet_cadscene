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

#version 450

#extension GL_GOOGLE_include_directive : enable
#extension GL_ARB_shading_language_include : enable
#include "config.h"

//////////////////////////////////////

#define USE_NATIVE   1
#extension GL_NV_mesh_shader : enable


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

/////////////////////////////////////////////////////////////////////////

#include "common.h"

/////////////////////////////////////////////////////////////////////////

#define GROUP_SIZE  WARP_SIZE

layout(local_size_x=GROUP_SIZE) in;

/////////////////////////////////////
// UNIFORMS

#if IS_VULKAN

  layout(push_constant) uniform pushConstant{
  #if !USE_PER_GEOMETRY_VIEWS
    uvec4     geometryOffsets;
  #endif
    uvec4     assigns;
  };
  #if USE_PER_GEOMETRY_VIEWS
    uvec4 geometryOffsets = uvec4(0, 0, 0, 0);
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

  layout(location = 1) uniform uvec4 assigns;

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

//////////////////////////////////////////////////////////////////////////
// INPUT

uint baseID = gl_WorkGroupID.x * GROUP_SIZE;
uint laneID = gl_LocalInvocationID.x;

//////////////////////////////////////////////////////////////////////////
// OUTPUT

taskNV out Task
{
  uint      baseID;
  uint8_t   subIDs[GROUP_SIZE];
} OUT;

//////////////////////////////////////////////////////////////////////////
// UTILS

#include "nvmeshlet_utils.glsl"

/////////////////////////////////////////////////
// EXECUTION

void main()
{
  baseID += assigns.x;
  uvec4 desc = meshletDescs[min(baseID + laneID, assigns.y) + geometryOffsets.x];

  bool render = !(baseID + laneID > assigns.y || earlyCull(desc, object));
  
#if IS_VULKAN
  uvec4 vote  = subgroupBallot(render);
  uint  tasks = subgroupBallotBitCount(vote);
  uint  voteGroup = vote.x;
#else
  uint vote = ballotThreadNV(render);
  uint tasks = bitCount(vote);
  uint voteGroup = vote;
#endif

  if (laneID == 0) {
    gl_TaskCountNV = tasks;
    OUT.baseID = baseID;
    #if USE_STATS
      atomicAdd(stats.tasksOutput, 1);
    #endif
  }

  {
  #if IS_VULKAN
    uint idxOffset = subgroupBallotExclusiveBitCount(vote);
  #else
    uint idxOffset = bitCount(vote & gl_ThreadLtMaskNV);
  #endif
    if (render) {
      OUT.subIDs[idxOffset] = uint8_t(laneID);
    }
  }
}