/*
 * Copyright (c) 2017-2022, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2017-2022 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */


//////////////////////////////////////////////////////////////////////////
// Affects both C++ and GLSL, must contain defines only

#ifndef _CONFIG_H_
#define _CONFIG_H_

/////////////////////////////////////////////////////

// keep in sync to nvmeshletbuilder.hpp PACKING configuration
// (values are explained there)

#define NVMESHLET_PRIM_ALIGNMENT    32
#define NVMESHLET_VERTEX_ALIGNMENT  16
#define NVMESHLET_BLOCK_ELEMENTS    32
#define NVMESHLET_PACK_ALIGNMENT    16
#define NVMESHLET_INDICES_PER_FETCH 8

#define NVMESHLET_ENCODING_PACKBASIC 1

#if !defined(__cplusplus)
#ifdef VULKAN
#define IS_VULKAN 1
#endif

#ifndef IS_VULKAN
#define IS_VULKAN 0
#endif
#endif

#if IS_VULKAN
  #define NVMESHLET_CLIP_Z_SIGNED 0
#else
  #define NVMESHLET_CLIP_Z_SIGNED 1
#endif

///////////////////////////////////////////////////

#if defined(__cplusplus)

  enum MeshletBuilderType {
    MESHLET_BUILDER_PACKBASIC,
  };

#endif

#endif
