/* Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

//////////////////////////////////////////////////////////////////////////
// Affects both C++ and GLSL, must contain defines only

#ifndef _CONFIG_H_
#define _CONFIG_H_

/////////////////////////////////////////////////////

// keep in sync to nvmeshletbuilder.hpp PACKING configuration
// (values are explained there)
// if set we use a tight or fitted packing, stored in RG32UI texture
// otherwise R32UI
#define NVMESHLET_PACKING_FITTED_UINT8   1
#define NVMESHLET_PRIM_ALIGNMENT        32
#define NVMESHLET_VERTEX_ALIGNMENT      16

#ifdef VULKAN 
#define IS_VULKAN 1
#endif

#ifndef IS_VULKAN
#define IS_VULKAN 0
#endif

#if IS_VULKAN && !defined(__cplusplus)
#extension GL_KHR_vulkan_glsl : enable
#endif

#if IS_VULKAN
  #define NVMESHLET_CLIP_Z_SIGNED 0
#else
  #define NVMESHLET_CLIP_Z_SIGNED 1
#endif

///////////////////////////////////////////////////

// must not change
#define WARP_SIZE  32
#define WARP_STEPS 5

///////////////////////////////////////////////////

// set to zero for less resources being generated
// also reduces runtime binding costs
#define USE_PER_GEOMETRY_VIEWS         0

#endif
