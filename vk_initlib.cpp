/*
 * Copyright (c) 2014-2021, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2014-2021 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */



#include <assert.h>
#include <string.h>
#include <vector>
#include <vulkan/vulkan_core.h>

#if HAS_OPENGL
#include <include_gl.h>
#include <nvgl/contextwindow_gl.hpp>
#endif

bool vulkanInitLibrary()
{
#if HAS_OPENGL
  if (!load_GL_NV_draw_vulkan_image(nvgl::ContextWindow::sysGetProcAddress)) return false;
#endif

#if NVP_SUPPORTS_VULKANSDK
  return true;
#else
  if (__nvkglGetVkProcAddrNV){
    vkLoadProcs( __nvkglGetVkProcAddrNV );
  }
  
  if (pfn_vkCreateDevice != NULL)
    return true;

  return false;
#endif
}

// non optimal brute force way
bool vulkanIsExtensionSupported(uint32_t deviceIdx, const char* name)
{
  VkResult result;

  VkInstance instance;
  VkInstanceCreateInfo instanceCreateInfo = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };


  result = vkCreateInstance(&instanceCreateInfo, NULL, &instance);
  if (result != VK_SUCCESS) {
    return false;
  }

  uint32_t physicalDeviceCount = 0;
  result = vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, NULL);

  if (result != VK_SUCCESS || physicalDeviceCount == 0) {
    return false;
  }

  std::vector<VkPhysicalDevice> physicalDevices(physicalDeviceCount);
  result = vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, physicalDevices.data());
  if (result != VK_SUCCESS) {
    vkDestroyInstance(instance, NULL);
    return false;
  }

  if (deviceIdx >= physicalDeviceCount) {
    return false;
  }

#ifndef NDEBUG
  std::vector<VkPhysicalDeviceProperties> physicalDeviceProperties(physicalDeviceCount);
  for (uint32_t i = 0; i < physicalDeviceCount; i++) {
    vkGetPhysicalDeviceProperties(physicalDevices[i], &physicalDeviceProperties[i]);
  }
#endif

  // pick first device
  VkPhysicalDevice physicalDevice = physicalDevices[deviceIdx];

  uint32_t count = 0;
  result = vkEnumerateDeviceExtensionProperties(physicalDevice, NULL, &count, NULL);
  if (result != VK_SUCCESS) {
    vkDestroyInstance(instance, NULL);
    return false;
  }

  std::vector<VkExtensionProperties> extensions(count);
  result = vkEnumerateDeviceExtensionProperties(physicalDevice, NULL, &count, extensions.data());
  if (result != VK_SUCCESS) {
    vkDestroyInstance(instance, NULL);
    return false;
  }
  
  bool found = false;

  for (uint32_t i = 0; i < count; i++){
    if (strcmp(extensions[i].extensionName, name) == 0){
      found = true;
      break;
    }
  }

  vkDestroyInstance(instance, NULL);

  return found;
}


