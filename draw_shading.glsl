/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

 
vec4 shading(vec3 wPos, vec3 wNormal, uint meshletID)
{  
  vec4 color = object.color * 0.8 + 0.2;
  if (scene.colorize != 0) {
    uint colorPacked = murmurHash(meshletID);
    color = color * 0.5 + unpackUnorm4x8(colorPacked) * 0.5;
  }
  
  vec3 eyePos = vec3(scene.viewMatrixIT[0].w,scene.viewMatrixIT[1].w,scene.viewMatrixIT[2].w);
  
  vec3 lightDir = normalize(scene.wLightPos.xyz - wPos.xyz);
  vec3 normal   = normalize(wNormal) * (gl_FrontFacing ? 1 : 1);

#if 1
  vec4 diffuse  = vec4(abs(dot(normal,lightDir)));
  vec4 outColor = diffuse * color;
#else
  float lt = abs(dot(normal,lightDir));
  float wt = dot(normal,lightDir) * 0.5 + 0.5;
  vec4 diffuse  = mix( pow((vec4(1)-color) * (1 - lt), vec4(0.7)), pow(color * lt * vec4(1,1,0.9,1), vec4(0.9)), pow(wt,0.5));
  vec4 outColor = diffuse;
#endif
  
  return outColor;
}