/*
 * Copyright (c) 2016-2021, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2016-2021 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
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