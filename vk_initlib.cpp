/* Copyright (c) 2014-2018, NVIDIA CORPORATION. All rights reserved.
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

#if USEVULKANSDK
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


