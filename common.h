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

/////////////////////////////////////////////////

// Input Mesh Vertex related

#define VERTEX_POS      0
#define VERTEX_NORMAL   1
#define VERTEX_EXTRAS   2  // must be NORMAL+1

#ifndef VERTEX_EXTRAS_COUNT
// add how many extra fake attributes (vec4) you want to use
#define VERTEX_EXTRAS_COUNT   4
#endif

#define VERTEX_NORMAL_STRIDE (1 + VERTEX_EXTRAS_COUNT)

/////////////////////////////////////////////////

// GL
#define UBO_SCENE_VIEW    0
#define UBO_OBJECT        1
#define UBO_GEOMETRY      2
#define SSBO_SCENE_STATS  0

// VK
#define DSET_SCENE        0
#define DSET_OBJECT       1
#define DSET_GEOMETRY     2

#define SCENE_UBO_VIEW    0
#define SCENE_SSBO_STATS  1

// changing order requires glsl changes in drawmesh_native.mesh.glsl
// geometryBuffer ubo
#define GEOMETRY_SSBO_MESHLETDESC   0
#define GEOMETRY_SSBO_PRIM          1
#define GEOMETRY_TEX_VBO            2
#define GEOMETRY_TEX_ABO            3
#define GEOMETRY_BINDINGS           5

////////////////////////////////////////////////////

#ifndef NVMESHLET_VERTEX_COUNT
// primitive count should be 40, 84 or 126
// vertex count should be 32 or 64
// 64 & 126 is the preferred size
#define NVMESHLET_VERTEX_COUNT      64
#define NVMESHLET_PRIMITIVE_COUNT   126
// must be multiple of SUBGROUP_SIZE
#define NVMESHLET_PER_TASK          32
#endif

#ifndef NVMESHLET_ENCODING
#define NVMESHLET_ENCODING  NVMESHLET_ENCODING_PACKBASIC
#endif

#ifndef MESH_SUBGROUP_SIZE
#define MESH_SUBGROUP_SIZE  32
#endif

#ifndef TASK_SUBGROUP_SIZE
#define TASK_SUBGROUP_SIZE  32
#endif

/////////////////////////////////////////////////


#ifndef USE_CLIPPING
#define USE_CLIPPING        0
#endif

#define NUM_CLIPPING_PLANES 3


/////////////////////////////////////////////////


#ifdef __cplusplus
namespace meshlettest
{
  using namespace nvmath;
#endif

struct SceneData {
  mat4  viewProjMatrix;
  mat4  viewMatrix;
  mat4  viewMatrixIT;

  vec4  viewPos;
  vec4  viewDir;
  
  vec4  wLightPos;
  
  ivec2 viewport;
  vec2  viewportf;

  vec2  viewportTaskCull;
  int   colorize;
  int   _pad0;
  
  vec4  wClipPlanes[NUM_CLIPPING_PLANES];
};

// must match cadscene!
struct ObjectData {
  mat4 worldMatrix;
  mat4 worldMatrixIT;
  mat4 objectMatrix;
  vec4 bboxMin;
  vec4 bboxMax;
  vec3 _pad0;
  float winding;
  vec4 color;
};

struct CullStats {
  uint  tasksInput;
  uint  tasksOutput;
  uint  meshletsInput;
  uint  meshletsOutput;
  uint  trisInput;
  uint  trisOutput;
  uint  attrInput;
  uint  attrOutput;
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
