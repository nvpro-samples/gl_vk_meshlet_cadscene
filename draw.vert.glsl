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

// We are using simple vertex attributes here, so
// that we can switch easily between fp32 and fp16 to
// investigate impact of vertex bandwith.
//
// In a more performance critical scenario we recommend the use
// of packed normals for CAD, like octant encoding and pack position
// and normal in a single 128-bit value.

in layout(location=VERTEX_POS)      vec3 oPos;
in layout(location=VERTEX_NORMAL)   vec3 oNormal;
#if VERTEX_EXTRAS_COUNT
in layout(location=VERTEX_EXTRAS)   vec4 xtra[VERTEX_EXTRAS_COUNT];
#endif

//////////////////////////////////////////////////
// OUTPUT

#if SHOW_PRIMIDS

  // nothing to output
  
#else

  layout(location=0) out Interpolants {
    vec3  wPos;    
    vec3  wNormal;
    flat uint meshletID;
  #if VERTEX_EXTRAS_COUNT
    vec4 xtra[VERTEX_EXTRAS_COUNT];
  #endif
} OUT;

#endif

#if IS_VULKAN && USE_CLIPPING
out float gl_ClipDistance[NUM_CLIPPING_PLANES];
#endif

//////////////////////////////////////////////////
// VERTEX EXECUTION

void main()
{
  vec3 wPos     = (object.worldMatrix  * vec4(oPos,1)).xyz;
  gl_Position   = (scene.viewProjMatrix * vec4(wPos,1));

  
#if USE_CLIPPING
  for (int i = 0; i < NUM_CLIPPING_PLANES; i++){
    gl_ClipDistance[i] = dot(scene.wClipPlanes[i], vec4(wPos,1));
  }
#endif
  
#if !SHOW_PRIMIDS
  vec3 wNormal  = mat3(object.worldMatrixIT) * oNormal;
  OUT.wNormal = wNormal;
  OUT.meshletID = 0; 
  OUT.wPos      = wPos;
  #if VERTEX_EXTRAS_COUNT
    UNROLL_LOOP
    for (int i = 0; i < VERTEX_EXTRAS_COUNT; i++){
      OUT.xtra[i] = xtra[i];
    }
  #endif
#endif
}
