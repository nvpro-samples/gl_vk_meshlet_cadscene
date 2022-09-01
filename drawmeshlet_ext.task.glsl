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


#version 460

  #extension GL_GOOGLE_include_directive : enable
  #extension GL_EXT_control_flow_attributes: require
  #define UNROLL_LOOP [[unroll]]


#include "config.h"

//////////////////////////////////////

  #extension GL_EXT_mesh_shader : require
  
//////////////////////////////////////

  #extension GL_EXT_shader_explicit_arithmetic_types_int8 : require

  #extension GL_KHR_shader_subgroup_basic : require
  #extension GL_KHR_shader_subgroup_ballot : require
  #extension GL_KHR_shader_subgroup_vote : require

/////////////////////////////////////////////////////////////////////////

#include "common.h"

/////////////////////////////////////////////////////////////////////////
// TASK CONFIG

// see Sample::getShaderPrepend() how these are computed
const uint WORKGROUP_SIZE = EXT_TASK_SUBGROUP_COUNT * EXT_TASK_SUBGROUP_SIZE;

layout(local_size_x=WORKGROUP_SIZE) in;

// The workgroup size of the shader may not have enough threads
// to do all the work in a unique thread.
// Therefore we might need to loop to process all the work.

const uint TASK_MESHLET_ITERATIONS = ((NVMESHLET_PER_TASK + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE);


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
  layout(std430, binding = GEOMETRY_SSBO_PRIM, set = DSET_GEOMETRY) buffer primIndexBuffer {
    uvec2 primIndices[];
  };

  layout(binding=GEOMETRY_TEX_VBO,  set=DSET_GEOMETRY)  uniform samplerBuffer  texVbo;
  layout(binding=GEOMETRY_TEX_ABO,  set=DSET_GEOMETRY)  uniform samplerBuffer  texAbo;

//////////////////////////////////////////////////////////////////////////
// INPUT

uint baseID = gl_WorkGroupID.x * NVMESHLET_PER_TASK;
uint laneID = gl_LocalInvocationID.x;

//////////////////////////////////////////////////////////////////////////
// OUTPUT


struct Task
{
  uint      baseID;
  uint8_t   deltaIDs[NVMESHLET_PER_TASK];
};

taskPayloadSharedEXT Task OUT;

//////////////////////////////////////////////////////////////////////////
// UTILS

#include "nvmeshlet_utils.glsl"

/////////////////////////////////////////////////
// EXECUTION

#define BARRIER() \
  memoryBarrierShared(); \
  barrier();

#if EXT_TASK_SUBGROUP_COUNT > 1
  shared uint s_outMeshletsCount;
#endif

void main()
{
#if EXT_TASK_SUBGROUP_COUNT > 1
  if (laneID == 0) {
    s_outMeshletsCount = 0;
  }
  BARRIER();
#endif

  baseID += drawRange.x;
  
  uint outMeshletsCount = 0;
  
  UNROLL_LOOP
  for (uint i = 0; i < TASK_MESHLET_ITERATIONS; i++)
  {
    uint  meshletLocal  = laneID + i * WORKGROUP_SIZE;
    uint  meshletGlobal = baseID + meshletLocal;
    uvec4 desc          = meshletDescs[min(meshletGlobal, drawRange.y) + geometryOffsets.x];
    
    bool render = !(meshletGlobal > drawRange.y || earlyCull(desc, object));

    uvec4 voteMeshlets = subgroupBallot(render);
    uint  numMeshlets  = subgroupBallotBitCount(voteMeshlets);
    
  #if EXT_TASK_SUBGROUP_COUNT > 1
    if (gl_SubgroupInvocationID == 0) {
      outMeshletsCount = atomicAdd(s_outMeshletsCount, numMeshlets);
    }
    outMeshletsCount = subgroupBroadcastFirst(outMeshletsCount);
  #endif
    
    uint idxOffset  = subgroupBallotExclusiveBitCount(voteMeshlets) + outMeshletsCount;
    if (render) 
    {
      OUT.deltaIDs[idxOffset] = uint8_t(meshletLocal);
    }
  #if EXT_TASK_SUBGROUP_COUNT == 1
    outMeshletsCount += numMeshlets;
  #endif
  }
  
#if EXT_TASK_SUBGROUP_COUNT > 1
  BARRIER();
  outMeshletsCount = s_outMeshletsCount;
#endif

  if (laneID == 0) {
    OUT.baseID = baseID;
  #if USE_STATS
    atomicAdd(stats.tasksOutput, outMeshletsCount);
  #endif
  }
  
  EmitMeshTasksEXT(outMeshletsCount, 1, 1);
}