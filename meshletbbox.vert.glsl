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

#if !IS_VULKAN
  #extension GL_NV_gpu_shader5 : require
  #extension GL_NV_bindless_texture : require
#endif

#include "common.h"

//////////////////////////////////////

#if IS_VULKAN

  #if USE_PER_GEOMETRY_VIEWS
    uvec4 geometryOffsets = uvec4(0, 0, 0, 0);
  #else
    layout(push_constant) uniform pushConstant{
      uvec4     geometryOffsets;
    };
  #endif

  layout(std140, binding = SCENE_UBO_VIEW, set = DSET_SCENE) uniform sceneBuffer {
    SceneData scene;
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

  layout(std140, binding = UBO_SCENE_VIEW) uniform sceneBuffer {
    SceneData scene;
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

//////////////////////////////////////

#include "nvmeshlet_utils.glsl"

//////////////////////////////////////

layout(location=0) out VertexOut{
  vec3 bboxCtr;
  vec3 bboxDim;
  vec3  coneNormal;
  float coneAngle;
  flat uint meshletID;
} OUT;

void main()
{
#if IS_VULKAN
  uint meshletID = uint(gl_VertexIndex);
#else
  uint meshletID = uint(gl_VertexID);
#endif
  uvec4 meshlet = meshletDescs[meshletID + geometryOffsets.x];

  vec3 bboxMin;
  vec3 bboxMax;
  decodeBbox(meshlet, object, bboxMin, bboxMax);
  
  vec3 ctr = (bboxMax + bboxMin) * 0.5;
  vec3 dim = (bboxMax - bboxMin) * 0.5;
    
  OUT.bboxCtr = ctr;
  OUT.bboxDim = dim;
  decodeNormalAngle(meshlet, object, OUT.coneNormal, OUT.coneAngle); 
  
  bool cull = earlyCull(meshlet, object);
#if SHOW_CULLED
  cull = !cull;
#endif
  
  if (cull) {
    OUT.meshletID = ~0u;
  }
  else {
    OUT.meshletID = meshletID;
  }
}
