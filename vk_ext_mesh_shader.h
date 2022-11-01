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
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <vulkan/vulkan_core.h>

// set to zero if nvvk::extensions_vk.cpp covers this
#define VK_EXT_mesh_shader_LOCAL 1

#ifndef VK_EXT_mesh_shader

#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT ((VkStructureType)1000328000)
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_PROPERTIES_EXT ((VkStructureType)1000328001)
#define VK_PIPELINE_STAGE_TASK_SHADER_BIT_EXT ((VkPipelineStageFlagBits)0x00080000)
#define VK_PIPELINE_STAGE_MESH_SHADER_BIT_EXT ((VkPipelineStageFlagBits)0x00100000)
#define VK_SHADER_STAGE_TASK_BIT_EXT ((VkShaderStageFlagBits)0x00000040)
#define VK_SHADER_STAGE_MESH_BIT_EXT ((VkShaderStageFlagBits)0x00000080)
#define VK_QUERY_TYPE_MESH_PRIMITIVES_GENERATED_EXT ((VkQueryType)1000328000)
#define VK_QUERY_PIPELINE_STATISTIC_TASK_SHADER_INVOCATIONS_BIT_EXT ((VkQueryPipelineStatisticFlagBits)0x00000800)
#define VK_QUERY_PIPELINE_STATISTIC_MESH_SHADER_INVOCATIONS_BIT_EXT ((VkQueryPipelineStatisticFlagBits)0x00001000)
#define VK_INDIRECT_COMMANDS_TOKEN_TYPE_DRAW_MESH_TASKS_NV ((VkIndirectCommandsTokenTypeNV)1000328000)
static const VkPipelineStageFlagBits2KHR VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT = 0x00080000ULL;
static const VkPipelineStageFlagBits2KHR VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT = 0x00100000ULL;

#define VK_EXT_mesh_shader 1
#define VK_EXT_MESH_SHADER_SPEC_VERSION 1
#define VK_EXT_MESH_SHADER_EXTENSION_NAME "VK_EXT_mesh_shader"
typedef struct VkPhysicalDeviceMeshShaderFeaturesEXT
{
  VkStructureType sType;
  void*           pNext;
  VkBool32        taskShader;
  VkBool32        meshShader;
  VkBool32        multiviewMeshShader;
  VkBool32        primitiveFragmentShadingRateMeshShader;
  VkBool32        meshShaderQueries;
} VkPhysicalDeviceMeshShaderFeaturesEXT;

typedef struct VkPhysicalDeviceMeshShaderPropertiesEXT
{
  VkStructureType sType;
  void*           pNext;
  uint32_t        maxTaskWorkGroupTotalCount;
  uint32_t        maxTaskWorkGroupCount[3];
  uint32_t        maxTaskWorkGroupInvocations;
  uint32_t        maxTaskWorkGroupSize[3];
  uint32_t        maxTaskPayloadSize;
  uint32_t        maxTaskSharedMemorySize;
  uint32_t        maxTaskPayloadAndSharedMemorySize;
  uint32_t        maxMeshWorkGroupTotalCount;
  uint32_t        maxMeshWorkGroupCount[3];
  uint32_t        maxMeshWorkGroupInvocations;
  uint32_t        maxMeshWorkGroupSize[3];
  uint32_t        maxMeshSharedMemorySize;
  uint32_t        maxMeshPayloadAndSharedMemorySize;
  uint32_t        maxMeshOutputMemorySize;
  uint32_t        maxMeshPayloadAndOutputMemorySize;
  uint32_t        maxMeshOutputComponents;
  uint32_t        maxMeshOutputVertices;
  uint32_t        maxMeshOutputPrimitives;
  uint32_t        maxMeshOutputLayers;
  uint32_t        maxMeshMultiviewViewCount;
  uint32_t        meshOutputPerVertexGranularity;
  uint32_t        meshOutputPerPrimitiveGranularity;
  uint32_t        maxPreferredTaskWorkGroupInvocations;
  uint32_t        maxPreferredMeshWorkGroupInvocations;
  VkBool32        prefersLocalInvocationVertexOutput;
  VkBool32        prefersLocalInvocationPrimitiveOutput;
  VkBool32        prefersCompactVertexOutput;
  VkBool32        prefersCompactPrimitiveOutput;
} VkPhysicalDeviceMeshShaderPropertiesEXT;

typedef struct VkDrawMeshTasksIndirectCommandEXT
{
  uint32_t groupCountX;
  uint32_t groupCountY;
  uint32_t groupCountZ;
} VkDrawMeshTasksIndirectCommandEXT;

typedef void(VKAPI_PTR* PFN_vkCmdDrawMeshTasksEXT)(VkCommandBuffer commandBuffer, uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ);
typedef void(VKAPI_PTR* PFN_vkCmdDrawMeshTasksIndirectEXT)(VkCommandBuffer commandBuffer,
                                                           VkBuffer        buffer,
                                                           VkDeviceSize    offset,
                                                           uint32_t        drawCount,
                                                           uint32_t        stride);
typedef void(VKAPI_PTR* PFN_vkCmdDrawMeshTasksIndirectCountEXT)(VkCommandBuffer commandBuffer,
                                                                VkBuffer        buffer,
                                                                VkDeviceSize    offset,
                                                                VkBuffer        countBuffer,
                                                                VkDeviceSize    countBufferOffset,
                                                                uint32_t        maxDrawCount,
                                                                uint32_t        stride);

VKAPI_ATTR void VKAPI_CALL vkCmdDrawMeshTasksEXT(VkCommandBuffer commandBuffer, uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ);

VKAPI_ATTR void VKAPI_CALL vkCmdDrawMeshTasksIndirectEXT(VkCommandBuffer commandBuffer,
                                                         VkBuffer        buffer,
                                                         VkDeviceSize    offset,
                                                         uint32_t        drawCount,
                                                         uint32_t        stride);

VKAPI_ATTR void VKAPI_CALL vkCmdDrawMeshTasksIndirectCountEXT(VkCommandBuffer commandBuffer,
                                                              VkBuffer        buffer,
                                                              VkDeviceSize    offset,
                                                              VkBuffer        countBuffer,
                                                              VkDeviceSize    countBufferOffset,
                                                              uint32_t        maxDrawCount,
                                                              uint32_t        stride);
#endif

int load_VK_EXT_mesh_shader(VkDevice device, PFN_vkGetDeviceProcAddr getDeviceProcAddr);
