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

in layout(location=VERTEX_POS)      vec3 oPos;
in layout(location=VERTEX_NORMAL)   vec3 oNormal;
#if EXTRA_ATTRIBUTES
in layout(location=VERTEX_XTRA)     vec4 xtra[EXTRA_ATTRIBUTES];
#endif

//////////////////////////////////////////////////
// OUTPUT

layout(location=0) out Interpolants {
  vec3  wPos;
  float dummy;
  vec3  wNormal;
  flat uint meshletID;
#if EXTRA_ATTRIBUTES
  vec4 xtra[EXTRA_ATTRIBUTES];
#endif
} OUT;

#if IS_VULKAN
out float gl_ClipDistance[NUM_CLIPPING_PLANES];
#endif

//////////////////////////////////////////////////
// EXECUTION

void main()
{
  vec3 wPos     = (object.worldMatrix  * vec4(oPos,1)).xyz;
  gl_Position   = (scene.viewProjMatrix * vec4(wPos,1));
  OUT.wPos      = wPos;
  OUT.dummy     = 0.0;
  
#if USE_CLIPPING
  for (int i = 0; i < NUM_CLIPPING_PLANES; i++){
    gl_ClipDistance[i] = dot(scene.wClipPlanes[i], vec4(wPos,1));
  }
#endif
  
  vec3 wNormal  = mat3(object.worldMatrixIT) * oNormal;
  OUT.wNormal = wNormal;
  OUT.meshletID = 0;
#if EXTRA_ATTRIBUTES
  for (int i = 0; i < EXTRA_ATTRIBUTES; i++){
    OUT.xtra[i] = xtra[i];
  }
#endif
}
