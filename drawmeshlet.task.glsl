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
  // one of them provides uint8_t
  #extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#else
  #extension GL_NV_gpu_shader5 : require
  #extension GL_NV_bindless_texture : require
#endif

  #extension GL_KHR_shader_subgroup_basic : require
  #extension GL_KHR_shader_subgroup_ballot : require
  #extension GL_KHR_shader_subgroup_vote : require

/////////////////////////////////////////////////////////////////////////

#include "common.h"

/////////////////////////////////////////////////////////////////////////

#define WORKGROUP_SIZE  TASK_SUBGROUP_SIZE

layout(local_size_x=WORKGROUP_SIZE) in;

/////////////////////////////////////
// UNIFORMS

#if IS_VULKAN

  layout(push_constant) uniform pushConstant{
    uvec4     geometryOffsets;
    uvec4     assigns;
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
  layout(std430, binding = GEOMETRY_SSBO_PRIM, set = DSET_GEOMETRY) buffer primIndexBuffer {
    uvec2 primIndices[];
  };

  layout(binding=GEOMETRY_TEX_VBO,  set=DSET_GEOMETRY)  uniform samplerBuffer  texVbo;
  layout(binding=GEOMETRY_TEX_ABO,  set=DSET_GEOMETRY)  uniform samplerBuffer  texAbo;

#else

  layout(location = 0) uniform uvec4 geometryOffsets;
  // x: mesh, y: prim, z: 0, w: vertex

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
    samplerBuffer   texVbo;
    samplerBuffer   texAbo;
  };
  
#endif

//////////////////////////////////////////////////////////////////////////
// INPUT

uint baseID = gl_WorkGroupID.x * NVMESHLET_PER_TASK;
uint laneID = gl_LocalInvocationID.x;

//////////////////////////////////////////////////////////////////////////
// OUTPUT

  // on NVIDIA hw the task-shader output should stay below
  // 108 bytes to stay on a very fast path. 236 bytes typically is
  // okay as well, but more is not recommended.
  
taskNV out Task
{
  uint      baseID;
  uint8_t   deltaIDs[NVMESHLET_PER_TASK];
} OUT;

//////////////////////////////////////////////////////////////////////////
// UTILS

#include "nvmeshlet_utils.glsl"

  // The workgroup size of the shader may not have enough threads
  // to do all the work in a unique thread.
  // Therefore we might need to loop to process all the work.

  #define TASK_MESHLET_ITERATIONS   ((NVMESHLET_PER_TASK + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)

/////////////////////////////////////////////////
// EXECUTION

void main()
{
  baseID += assigns.x;
  
  uint numOutTasks = 0;
  
  UNROLL_LOOP
  for (uint i = 0; i < TASK_MESHLET_ITERATIONS; i++)
  {
    uint  meshletLocal  = laneID + i * WORKGROUP_SIZE;
    uint  meshletGlobal = baseID + meshletLocal;
    uvec4 desc          = meshletDescs[min(meshletGlobal, assigns.y) + geometryOffsets.x];
    
    bool render = !(meshletGlobal > assigns.y || earlyCull(desc, object));

    uvec4 voteTasks = subgroupBallot(render);
    uint  numTasks  = subgroupBallotBitCount(voteTasks);
    uint idxOffset  = subgroupBallotExclusiveBitCount(voteTasks) + numOutTasks;
    if (render) 
    {
      OUT.deltaIDs[idxOffset] = uint8_t(meshletLocal);
    }
    
    numOutTasks += numTasks;
  }

  if (laneID == 0) {
    gl_TaskCountNV = numOutTasks;
    OUT.baseID     = baseID;
    #if USE_STATS
      atomicAdd(stats.tasksOutput, numOutTasks);
    #endif
  }
}