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
#else
#extension GL_ARB_shading_language_include : enable
#endif

#include "common.h"

layout(location=0,index=0) out vec4 out_Color;

layout(location=0) in flat uint meshletID;

void main (){
  uint cp = murmurHash(meshletID);
  out_Color = unpackUnorm4x8(cp);
}