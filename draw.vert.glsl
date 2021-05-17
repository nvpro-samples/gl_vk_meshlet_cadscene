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
