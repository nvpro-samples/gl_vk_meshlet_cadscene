/*
 * Copyright (c) 2016-2023, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2016-2023 NVIDIA CORPORATION
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
  #if USE_BARYCENTRIC_SHADING_QUADSHUFFLE
    #extension GL_KHR_shader_subgroup_basic : require
    #extension GL_KHR_shader_subgroup_quad  : require
  #endif

  #if USE_BARYCENTRIC_SHADING_EXT
    #extension GL_EXT_fragment_shader_barycentric : require
  #else
    #extension GL_NV_fragment_shader_barycentric : require
    #define pervertexEXT    pervertexNV
    #define gl_BaryCoordEXT gl_BaryCoordNV
  #endif
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
  
  layout(location=1) pervertexEXT in ManualInterpolants {
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

  // With barycentric shading we move per-vertex work into the
  // fragment shader. We use the builtin gl_BaryCoordEXT to interpolate
  // the per-vertex values for fragment shading.
  //
  // One motivation for doing this in a scenario with lots of tiny
  // triangles on screen is that the pre-raster stages (vertex or mesh shaders) 
  // can become occupancy limited by their per-vertex outputs. These outputs
  // become less utilized if the majority of triangles are not rastered (too small etc.).
  // We essentially waste computing vertices not visible, and because
  // hw had to pre-allocate them, we reduced occupancy of those
  // vertex and mesh shader warps the more vertex attributes we pass to
  // fragment shader.


#if USE_BARYCENTRIC_SHADING_QUADSHUFFLE

  // One downside of doing per-vertex work in the fragment shader is that
  // (like in raytracing) we have to do all the per-vertex work in just
  // one fragment shader thread, trippling the work.
  // However, as we actually shade in quads, we can distribute each of the
  // three vertices for the incoming triangle within the quad into a separate thread, 
  // and as result regain some thread effiency.
  //
  // This sample's per-vertex work is too simple to really show much of a benefit.
  // You have to increase the "extra vertex attributes" in the UI quite some.
  //
  // However, in case your data ends up being rendered as tiny triangles and your code 
  // does some more complex per-vertex work, then this technique can help overall 
  // performance.
  
  uint quadIndex  = gl_SubgroupInvocationID & 3;
  uint vidx       = INBary[min(quadIndex,3)].vidx;
  

  vec3 oPos = getPosition(vidx);
  vec3 wPos = (mat4(object.worldMatrix) * vec4(oPos,1)).xyz;
  
  vec3 oNormal = getNormal(vidx);
  vec3 wNormal  = mat3(object.worldMatrixIT) * oNormal;
  
  wPos = subgroupQuadBroadcast(wPos, 0) * gl_BaryCoordEXT.x +
         subgroupQuadBroadcast(wPos, 1) * gl_BaryCoordEXT.y +
         subgroupQuadBroadcast(wPos, 2) * gl_BaryCoordEXT.z;
         
  wNormal = subgroupQuadBroadcast(wNormal, 0) * gl_BaryCoordEXT.x +
            subgroupQuadBroadcast(wNormal, 1) * gl_BaryCoordEXT.y +
            subgroupQuadBroadcast(wNormal, 2) * gl_BaryCoordEXT.z;

  vec4 color = shading(wPos, wNormal, 0);
  #if VERTEX_EXTRAS_COUNT
  {
    vec4 xtra = vec4(0);
    UNROLL_LOOP
    for (int i = 0; i < VERTEX_EXTRAS_COUNT; i++){
      xtra += getExtra(vidx, i);
    }
    
    xtra = subgroupQuadBroadcast(xtra, 0) * gl_BaryCoordEXT.x +
           subgroupQuadBroadcast(xtra, 1) * gl_BaryCoordEXT.y +
           subgroupQuadBroadcast(xtra, 2) * gl_BaryCoordEXT.z;
    
    color += xtra;
  }
  #endif
  out_Color = color;
  
#else

  // without quad shuffle, we do each per-vertex work in the same thread

  vec3 oPos = getPosition(INBary[0].vidx) * gl_BaryCoordEXT.x 
            + getPosition(INBary[1].vidx) * gl_BaryCoordEXT.y 
            + getPosition(INBary[2].vidx) * gl_BaryCoordEXT.z;
  vec3 wPos = (mat4(object.worldMatrix) * vec4(oPos,1)).xyz;
  
  vec3 oNormal = getNormal(INBary[0].vidx) * gl_BaryCoordEXT.x 
               + getNormal(INBary[1].vidx) * gl_BaryCoordEXT.y 
               + getNormal(INBary[2].vidx) * gl_BaryCoordEXT.z;
  vec3 wNormal  = mat3(object.worldMatrixIT) * oNormal;

  vec4 color = shading(wPos, wNormal, 0);
  #if VERTEX_EXTRAS_COUNT
  {
    UNROLL_LOOP
    for (int i = 0; i < VERTEX_EXTRAS_COUNT; i++){
      vec4 xtra = getExtra(INBary[0].vidx, i) * gl_BaryCoordEXT.x 
                + getExtra(INBary[1].vidx, i) * gl_BaryCoordEXT.y 
                + getExtra(INBary[2].vidx, i) * gl_BaryCoordEXT.z;
      color += xtra;
    }
  }
  #endif
  out_Color = color;

#endif
  
#else
  
  // In the traditional way, without fragment shader barycentrics,
  // interpolated values are used directly

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
