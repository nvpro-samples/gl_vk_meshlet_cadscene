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
  
  #extension GL_NV_gpu_shader5 : require
  #extension GL_NV_bindless_texture : require
#endif


#if USE_BARYCENTRIC_SHADING
  #extension GL_NV_fragment_shader_barycentric : require
#endif

#include "common.h"

//////////////////////////////////////////////////
// UNIFORMS

#if IS_VULKAN

  layout(std140,binding= SCENE_UBO_VIEW,set=DSET_SCENE) uniform sceneBuffer {
    SceneData scene;
  };

  layout(std140,binding=0,set=DSET_OBJECT) uniform objectBuffer {
    ObjectData object;
  };
  
  #if USE_BARYCENTRIC_SHADING
  layout(std430, binding = GEOMETRY_SSBO_MESHLETDESC, set = DSET_GEOMETRY) buffer meshletDescBuffer {
    uvec4 meshletDescs[];
  };
  layout(std430, binding = GEOMETRY_SSBO_PRIM, set = DSET_GEOMETRY) buffer primIndexBuffer1 {
    uint  primIndices1[];
  };
  layout(std430, binding = GEOMETRY_SSBO_PRIM, set = DSET_GEOMETRY) buffer primIndexBuffer2 {
    uvec2 primIndices2[];
  };

  layout(binding=GEOMETRY_TEX_VBO,  set=DSET_GEOMETRY)  uniform samplerBuffer  texVbo;
  layout(binding=GEOMETRY_TEX_ABO,  set=DSET_GEOMETRY)  uniform samplerBuffer  texAbo;
  #endif
  
#else

  layout(std140,binding=UBO_SCENE_VIEW) uniform sceneBuffer {
    SceneData scene;
  };

  layout(std140,binding=UBO_OBJECT) uniform objectBuffer {
    ObjectData object;
  };
  
  #if USE_BARYCENTRIC_SHADING
  // keep in sync with binding order defined via GEOMETRY_
  layout(std140, binding = UBO_GEOMETRY) uniform geometryBuffer{
    uvec4*          meshletDescs;
    uvec2*          primIndices;
    samplerBuffer   texVbo;
    samplerBuffer   texAbo;
  };
  #endif

#endif

//////////////////////////////////////////////////
// INPUT

#if SHOW_PRIMIDS

  // no inputs

#elif USE_BARYCENTRIC_SHADING

  layout(location=0) in Interpolants {
    flat uint meshletID;
  } IN;
  
  layout(location=1) pervertexNV in ManualInterpolants {
    uint vidx;
  } INBary[3];

#else

  layout(location=0) in Interpolants {
    vec3  wPos;
    vec3  wNormal;
    flat uint meshletID;
  #if VERTEX_EXTRAS_COUNT
    vec4 xtra[VERTEX_EXTRAS_COUNT];
  #endif
  } IN;

#endif

//////////////////////////////////////////////////
// OUTPUT

layout(location=0,index=0) out vec4 out_Color;


//////////////////////////////////////////////////
// EXECUTION

#if !SHOW_PRIMIDS
#include "draw_shading.glsl"
#endif

#if USE_BARYCENTRIC_SHADING

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

#endif

void main()
{
#if SHOW_PRIMIDS

  uint colorPacked = murmurHash(gl_PrimitiveID);
  out_Color = unpackUnorm4x8(colorPacked);
  
#elif USE_BARYCENTRIC_SHADING

  vec3 oPos = getPosition(INBary[0].vidx) * gl_BaryCoordNV.x + getPosition(INBary[1].vidx) * gl_BaryCoordNV.y + getPosition(INBary[2].vidx) * gl_BaryCoordNV.z;
  vec3 wPos = (mat4(object.worldMatrix) * vec4(oPos,1)).xyz;
  
  vec3 oNormal = getNormal(INBary[0].vidx) * gl_BaryCoordNV.x + getNormal(INBary[1].vidx) * gl_BaryCoordNV.y + getNormal(INBary[2].vidx) * gl_BaryCoordNV.z;
  vec3 wNormal  = mat3(object.worldMatrixIT) * oNormal;

  vec4 color = shading(wPos, wNormal, 0);
  #if VERTEX_EXTRAS_COUNT
  {
    UNROLL_LOOP
    for (int i = 0; i < VERTEX_EXTRAS_COUNT; i++){
      vec4 xtra = getExtra(INBary[0].vidx, i) * gl_BaryCoordNV.x + getExtra(INBary[1].vidx, i) * gl_BaryCoordNV.y + getExtra(INBary[2].vidx, i) * gl_BaryCoordNV.z;
      color += xtra;
    }
  }
  #endif
  out_Color = color;
  
#else

  vec4 color = shading(IN.wPos, IN.wNormal, IN.meshletID);
  #if VERTEX_EXTRAS_COUNT
  {
    UNROLL_LOOP
    for (int i = 0; i < VERTEX_EXTRAS_COUNT; i++){
      color += IN.xtra[i];
    }
  }
  #endif
  out_Color = color;
  
#endif
}
