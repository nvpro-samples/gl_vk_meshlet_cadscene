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

#include "vk_ext_mesh_shader.h"

static PFN_vkCmdDrawMeshTasksIndirectCountEXT pfn_vkCmdDrawMeshTasksIndirectCountEXT = nullptr;
static PFN_vkCmdDrawMeshTasksIndirectEXT      pfn_vkCmdDrawMeshTasksIndirectEXT      = nullptr;
static PFN_vkCmdDrawMeshTasksEXT              pfn_vkCmdDrawMeshTasksEXT              = nullptr;

#if VK_EXT_mesh_shader_LOCAL
VKAPI_ATTR void VKAPI_CALL vkCmdDrawMeshTasksIndirectCountEXT(VkCommandBuffer commandBuffer,
                                                             VkBuffer        buffer,
                                                             VkDeviceSize    offset,
                                                             VkBuffer        countBuffer,
                                                             VkDeviceSize    countBufferOffset,
                                                             uint32_t        maxDrawCount,
                                                             uint32_t        stride)
{
  pfn_vkCmdDrawMeshTasksIndirectCountEXT(commandBuffer, buffer, offset, countBuffer, countBufferOffset, maxDrawCount, stride);
}
VKAPI_ATTR void VKAPI_CALL
vkCmdDrawMeshTasksIndirectEXT(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset, uint32_t drawCount, uint32_t stride)
{
  pfn_vkCmdDrawMeshTasksIndirectEXT(commandBuffer, buffer, offset, drawCount, stride);
}
VKAPI_ATTR void VKAPI_CALL vkCmdDrawMeshTasksEXT(VkCommandBuffer commandBuffer, uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ)
{
  pfn_vkCmdDrawMeshTasksEXT(commandBuffer, groupCountX, groupCountY, groupCountZ);
}
#endif

int load_VK_EXT_mesh_shader(VkDevice device, PFN_vkGetDeviceProcAddr getDeviceProcAddr)
{
  pfn_vkCmdDrawMeshTasksIndirectCountEXT =
      (PFN_vkCmdDrawMeshTasksIndirectCountEXT)getDeviceProcAddr(device, "vkCmdDrawMeshTasksIndirectCountEXT");
  pfn_vkCmdDrawMeshTasksIndirectEXT =
      (PFN_vkCmdDrawMeshTasksIndirectEXT)getDeviceProcAddr(device, "vkCmdDrawMeshTasksIndirectEXT");
  pfn_vkCmdDrawMeshTasksEXT = (PFN_vkCmdDrawMeshTasksEXT)getDeviceProcAddr(device, "vkCmdDrawMeshTasksEXT");

  return pfn_vkCmdDrawMeshTasksIndirectCountEXT != nullptr && pfn_vkCmdDrawMeshTasksIndirectEXT != nullptr
         && pfn_vkCmdDrawMeshTasksEXT != nullptr;
}
