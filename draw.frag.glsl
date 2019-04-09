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
/**/

#extension GL_ARB_shading_language_include : enable
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
  
#else

  layout(std140,binding=UBO_SCENE_VIEW) uniform sceneBuffer {
    SceneData scene;
  };

  layout(std140,binding=UBO_OBJECT) uniform objectBuffer {
    ObjectData object;
  };

#endif

//////////////////////////////////////////////////
// INPUT

layout(location=0) in Interpolants {
  vec3  wPos;
  float dummy;
  vec3  wNormal;
  flat uint meshletID;
#if EXTRA_ATTRIBUTES
  vec4 xtra[EXTRA_ATTRIBUTES];
#endif
} IN;

//////////////////////////////////////////////////
// OUTPUT

layout(location=0,index=0) out vec4 out_Color;

//////////////////////////////////////////////////
// EXECUTION

void main()
{
  vec4 color = object.color * 0.8 + 0.2 + IN.dummy;
  if (scene.colorize != 0) {
    uint colorPacked = murmurHash(IN.meshletID);
    color = color * 0.5 + unpackUnorm4x8(colorPacked) * 0.5;
  }
  
  vec3 eyePos = vec3(scene.viewMatrixIT[0].w,scene.viewMatrixIT[1].w,scene.viewMatrixIT[2].w);
  
  vec3 wNormal =  IN.wNormal;
  vec3 lightDir = normalize(scene.wLightPos.xyz - IN.wPos.xyz);
  vec3 normal   = normalize(wNormal) * (gl_FrontFacing ? 1 : 1);

#if 1
  vec4 diffuse  = vec4(abs(dot(normal,lightDir)));
  out_Color = diffuse * color;
#else
  float lt = abs(dot(normal,lightDir));
  float wt = dot(normal,lightDir) * 0.5 + 0.5;
  vec4 diffuse  = mix( pow((vec4(1)-color) * (1 - lt), vec4(0.7)), pow(color * lt * vec4(1,1,0.9,1), vec4(0.9)), pow(wt,0.5));
  out_Color = diffuse;
#endif

  
  
  #if EXTRA_ATTRIBUTES
    for (int i = 0; i < EXTRA_ATTRIBUTES; i++){
      out_Color += IN.xtra[i];
    }
  #endif
}
