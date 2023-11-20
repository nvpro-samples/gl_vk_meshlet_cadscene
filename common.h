/*
 * Copyright (c) 2017-2022, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2017-2022 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */


#ifndef _COMMON_H_
#define _COMMON_H_

#include "config.h"


////////////////////////////////////////////////////
////////////////////////////////////////////////////
// Shader Configuration
//
// set in Sample::getShaderPrepend()
// set some defaults here for sake of offline compilation
// actual values always will come from app

#ifndef NVMESHLET_VERTEX_COUNT
// primitive count should be 40, 84 or 126
//    vertex count should be 32 or 64
//
// 64 vertices &  84 triangles:
//    works typically well for NV
// 64 vertices &  64 triangles:
//    is more portable for EXT usage
//    (hw that does 128 & 128 well, can do 2 x 64 & 64 at once)
// 64 vertices & 126 triangles:
//    can work in z-only or other very low extra
//    vertex attribute scenarios for NV
//
#define NVMESHLET_VERTEX_COUNT 64
#define NVMESHLET_PRIMITIVE_COUNT 64
// must be multiple of SUBGROUP_SIZE
#define NVMESHLET_PER_TASK 32
#endif

#ifndef NVMESHLET_ENCODING
#define NVMESHLET_ENCODING NVMESHLET_ENCODING_PACKBASIC
#endif

/////////////////////////////////////////////////
// EXT_mesh_shader preferences
//
// set in Sample::getShaderPrepend()
// these values reflect those found in VkPhysicalDeviceMeshShaderPropertiesEXT

// use a 1:1 mapping between gl_LocalInvocationIndex and output vertex index
// e.g. all writes should effectively be
// gl_MeshVerticesEXT[gl_LocalInvocationIndex] = ...
#ifndef EXT_LOCAL_INVOCATION_VERTEX_OUTPUT
#define EXT_LOCAL_INVOCATION_VERTEX_OUTPUT 0
#endif

// use a 1:1 mapping between gl_LocalInvocationIndex and output primitive index
// e.g. all writes should effectively be
// gl_MeshPrimitivesEXT[gl_LocalInvocationIndex] = ...
// gl_PrimitiveTriangleIndicesEXT[gl_LocalInvocationIndex] = ...
#ifndef EXT_LOCAL_INVOCATION_PRIMITIVE_OUTPUT
#define EXT_LOCAL_INVOCATION_PRIMITIVE_OUTPUT 0
#endif

// In a scenario where primitive culling and vertex culling
// is used, only the surviving vertices should be allocated
// and written as output.
// Otherwise, it is okay just output all vertices (interleaved
// used and unused) and leave the removal of unused to hardware.
// Compacting the vertex outputs also means the primitive
// indices have to be re-indexed.
#ifndef EXT_COMPACT_VERTEX_OUTPUT
#define EXT_COMPACT_VERTEX_OUTPUT 0
#endif

// In a scenario where primitive culling is used
// only those triangles that survive should be allocated
// and written as output.
// Otherwise, it is okay just output all primitives (interleaved
// used and unused) and leave the removal of unused to hardware.
// Use degenerate or ideally gl_CullPrimitiveEXT to mark culled
// primitives.
#ifndef EXT_COMPACT_PRIMITIVE_OUTPUT
#define EXT_COMPACT_PRIMITIVE_OUTPUT 0
#endif

// hardware specific preferences
#ifndef EXT_MAX_MESH_WORKGROUP_INVOCATIONS
#define EXT_MAX_MESH_WORKGROUP_INVOCATIONS 32
#endif

#ifndef EXT_MAX_TASK_WORKGROUP_INVOCATIONS
#define EXT_MAX_TASK_WORKGROUP_INVOCATIONS 32
#endif

#ifndef EXT_MESH_SUBGROUP_SIZE
#define EXT_MESH_SUBGROUP_SIZE 32
#endif

#ifndef EXT_TASK_SUBGROUP_SIZE
#define EXT_TASK_SUBGROUP_SIZE 32
#endif

// We want to maximize the number of available threads to ideally
// get a 1:1 mapping for outputs, and then align to subgroups.
//
// EXT_MESH_SUBGROUP_COUNT =
//   (min(max(NVMESHLET_VERTEX_COUNT, NVMESHLET_PRIMITIVE_COUNT), EXT_MAX_MESH_WORKGROUP_INVOCATIONS)
//   + EXT_MESH_SUBGROUP_SIZE - 1) / EXT_MESH_SUBGROUP_SIZE
#ifndef EXT_MESH_SUBGROUP_COUNT
#define EXT_MESH_SUBGROUP_COUNT 1
#endif

// for task shaders we maximize threads to fit NVMESHLET_PER_TASK
//
// EXT_TASK_SUBGROUP_COUNT =
//   (min(NVMESHLET_PER_TASK, EXT_MAX_TASK_WORKGROUP_INVOCATIONS)
//   + EXT_TASK_SUBGROUP_SIZE - 1) / EXT_TASK_SUBGROUP_SIZE
#ifndef EXT_TASK_SUBGROUP_COUNT
#define EXT_TASK_SUBGROUP_COUNT 1
#endif

/////////////////////////////////////////////////
// set in Sample::getShaderPrepend()

#ifndef USE_BARYCENTRIC_SHADING
#define USE_BARYCENTRIC_SHADING 0
#endif

#ifndef USE_BACKFACECULL
#define USE_BACKFACECULL 1
#endif

#ifndef USE_CLIPPING
#define USE_CLIPPING 0
#endif

#ifndef USE_STATS
#define USE_STATS 0
#endif

#ifndef SHOW_PRIMIDS
#define SHOW_PRIMIDS 0
#endif

#ifndef SHOW_BOX
#define SHOW_BOX 0
#endif

#ifndef SHOW_NORMAL
#define SHOW_NORMAL 0
#endif

#ifndef SHOW_CULLED
#define SHOW_CULLED 0
#endif


////////////////////////////////////////////////////
////////////////////////////////////////////////////

// Misc settings

// in our sample near/far clipping doesn't really apply much so
// this as disabled for now
#define USE_CULLBITS  0


// Input Mesh Vertex related

#define VERTEX_POS 0
#define VERTEX_NORMAL 1
#define VERTEX_EXTRAS 2  // must be NORMAL+1

#ifndef VERTEX_EXTRAS_COUNT
// add how many extra fake attributes (vec4) you want to use
#define VERTEX_EXTRAS_COUNT 1
#endif
#define VERTEX_NORMAL_STRIDE (1 + VERTEX_EXTRAS_COUNT)


#define NUM_CLIPPING_PLANES 3

/////////////////////////////////////////////////
// Binding Slots

// GL
#define UBO_SCENE_VIEW 0
#define UBO_OBJECT 1
#define UBO_GEOMETRY 2
#define SSBO_SCENE_STATS 0

// VK
#define DSET_SCENE 0
#define DSET_OBJECT 1
#define DSET_GEOMETRY 2

#define SCENE_UBO_VIEW 0
#define SCENE_SSBO_STATS 1

// changing order requires glsl changes in drawmesh_native.mesh.glsl
// geometryBuffer ubo
#define GEOMETRY_SSBO_MESHLETDESC 0
#define GEOMETRY_SSBO_PRIM 1
#define GEOMETRY_TEX_VBO 2
#define GEOMETRY_TEX_ABO 3
#define GEOMETRY_BINDINGS 5

/////////////////////////////////////////////////


#ifdef __cplusplus
namespace meshlettest {
using namespace glm;
#endif

struct SceneData
{
  mat4 viewProjMatrix;
  mat4 viewMatrix;
  mat4 viewMatrixIT;

  vec4 viewPos;
  vec4 viewDir;

  vec4 wLightPos;

  ivec2 viewport;
  vec2  viewportf;

  vec2 viewportTaskCull;
  int  colorize;
  int  _pad0;

  vec4 wClipPlanes[NUM_CLIPPING_PLANES];
};

// must match cadscene!
struct ObjectData
{
  mat4  worldMatrix;
  mat4  worldMatrixIT;
  mat4  objectMatrix;
  vec4  bboxMin;
  vec4  bboxMax;
  vec3  _pad0;
  float winding;
  vec4  color;
};

struct CullStats
{
  uint tasksInput;
  uint tasksOutput;
  uint meshletsInput;
  uint meshletsOutput;
  uint trisInput;
  uint trisOutput;
  uint attrInput;
  uint attrOutput;

  uint debugA[64];
  uint debugB[64];
  uint debugC[64];
};


#ifdef __cplusplus
}
#else

// GLSL functions

uint murmurHash(uint idx)
{
  uint m = 0x5bd1e995;
  uint r = 24;

  uint h = 64684;
  uint k = idx;

  k *= m;
  k ^= (k >> r);
  k *= m;
  h *= m;
  h ^= k;

  return h;
}
#endif

#endif
