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

#if SHOW_PRIMIDS

  // no inputs

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

void main()
{
#if SHOW_PRIMIDS

  uint colorPacked = murmurHash(gl_PrimitiveID);
  out_Color = unpackUnorm4x8(colorPacked);
  
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
