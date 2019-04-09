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
