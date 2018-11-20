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
