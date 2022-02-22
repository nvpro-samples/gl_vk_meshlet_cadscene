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
#else
#extension GL_ARB_shading_language_include : enable
#endif

#include "common.h"

////////////////////////////////////////////////

#if IS_VULKAN

  layout(std140, binding = SCENE_UBO_VIEW, set = DSET_SCENE) uniform sceneBuffer {
    SceneData scene;
  };

  layout(std140, binding = 0, set = DSET_OBJECT) uniform objectBuffer {
    ObjectData object;
  };

#else

  layout(std140, binding = UBO_SCENE_VIEW) uniform sceneBuffer {
    SceneData scene;
  };

  layout(std140, binding = UBO_OBJECT) uniform objectBuffer {
    ObjectData object;
  };

#endif

/////////////////////////////////////////

#define BOX_SIDES     6

#ifndef SHOW_BOX
#define SHOW_BOX      1
#endif

#ifndef SHOW_NORMAL
#define SHOW_NORMAL   0
#endif

// render the 6 visible sides based on view direction and box normal
#if SHOW_NORMAL && SHOW_BOX
layout(points,invocations=7) in;  
#elif SHOW_BOX
layout(points,invocations=6) in;  
#else
layout(points,invocations=1) in;  
#endif

// one side each invocation
layout(triangle_strip,max_vertices=4) out;

layout(location=0) in VertexOut{
  vec3 bboxCtr;
  vec3 bboxDim;
  vec3  coneNormal;
  float coneAngle;
  flat uint meshletID;
} IN[1];

layout(location=0) out flat uint meshletID;

void main()
{
  //bool skip = scene.filterID >= 0 && objid != scene.filterID;
  //if (skip) return;
  
  if (IN[0].meshletID == ~0u) return;

  mat4 worldTM  = object.worldMatrix;
  vec3 worldCtr = (worldTM * vec4(IN[0].bboxCtr, 1)).xyz;
  
  vec3 faceNormal = vec3(0);
  vec3 edgeBasis0 = vec3(0);
  vec3 edgeBasis1 = vec3(0);
  
  int id = gl_InvocationID;
  id = id % 3;

  if (id == 0)
  {
      faceNormal.x = IN[0].bboxDim.x;
      edgeBasis0.y = IN[0].bboxDim.y;
      edgeBasis1.z = IN[0].bboxDim.z;
  }
  else if(id == 1)
  {
      faceNormal.y = IN[0].bboxDim.y;
      edgeBasis1.x = IN[0].bboxDim.x;
      edgeBasis0.z = IN[0].bboxDim.z;
  }
  else if(id == 2)
  {
      faceNormal.z = IN[0].bboxDim.z;
      edgeBasis0.x = IN[0].bboxDim.x;
      edgeBasis1.y = IN[0].bboxDim.y;
  }


  vec3 worldNormal = mat3(worldTM) * faceNormal;
  vec3 worldPos    = worldCtr + worldNormal;
  float proj = -sign(dot(worldPos - scene.viewPos.xyz, worldNormal));

  if (gl_InvocationID > 2) {
    proj = -proj;
  }
    
  faceNormal = mat3(worldTM) * (faceNormal);
  edgeBasis0 = mat3(worldTM) * (edgeBasis0);
  edgeBasis1 = mat3(worldTM) * (edgeBasis1);

  faceNormal *= proj;
  edgeBasis1 *= proj;
  
#if SHOW_NORMAL || (!SHOW_BOX)
  #if SHOW_BOX
  if (gl_InvocationID >= BOX_SIDES) 
  #endif
  {
    // cone triangle
    
    meshletID = IN[0].meshletID;
    gl_Position = scene.viewProjMatrix * vec4(worldCtr, 1);
    EmitVertex();
    
    meshletID = IN[0].meshletID;
    gl_Position = scene.viewProjMatrix * vec4(worldCtr + (edgeBasis0) * 0.1,1);
    EmitVertex();
    
    vec3 worldDir = normalize(mat3(object.worldMatrixIT) * IN[0].coneNormal) * object.winding;
    float len = (length(faceNormal) + length(edgeBasis0) + length(edgeBasis1)) / 3.0;
    len *= IN[0].coneAngle > 0 ? 0.01 : 1.0;
    
    meshletID = IN[0].meshletID;
    gl_Position = scene.viewProjMatrix * vec4(worldCtr + worldDir * len,1);
    EmitVertex();
    
    meshletID = IN[0].meshletID;
    gl_Position = scene.viewProjMatrix * vec4(worldCtr + (edgeBasis1) * 0.1,1);
    EmitVertex();
  }
#endif
#if SHOW_BOX
  #if SHOW_NORMAL
  else
  #endif
  {
    meshletID = IN[0].meshletID;
    gl_Position = scene.viewProjMatrix * vec4(worldCtr + (faceNormal - edgeBasis0 - edgeBasis1),1);
    EmitVertex();
    
    meshletID = IN[0].meshletID;
    gl_Position = scene.viewProjMatrix * vec4(worldCtr + (faceNormal + edgeBasis0 - edgeBasis1),1);
    EmitVertex();
    
    meshletID = IN[0].meshletID;
    gl_Position = scene.viewProjMatrix * vec4(worldCtr + (faceNormal - edgeBasis0 + edgeBasis1),1);
    EmitVertex();
    
    meshletID = IN[0].meshletID;
    gl_Position = scene.viewProjMatrix * vec4(worldCtr + (faceNormal + edgeBasis0 + edgeBasis1),1);
    EmitVertex();
  }
#endif
}
