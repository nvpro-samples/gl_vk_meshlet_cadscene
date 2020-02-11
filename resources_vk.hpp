/* Copyright (c) 2016-2018, NVIDIA CORPORATION. All rights reserved.
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


#pragma once

#include "cadscene_vk.hpp"
#include "resources.hpp"

#include <nvvk/context_vk.hpp>
#include <nvvk/profiler_vk.hpp>
#include <nvvk/shadermodulemanager_vk.hpp>
#include <nvvk/swapchain_vk.hpp>

#include <nvvk/buffers_vk.hpp>
#include <nvvk/commands_vk.hpp>
#include <nvvk/descriptorsets_vk.hpp>
#include <nvvk/error_vk.hpp>
#include <nvvk/memorymanagement_vk.hpp>
#include <nvvk/renderpasses_vk.hpp>

class NVPWindow;

#define DSET_COUNT 3

namespace meshlettest {

class ResourcesVK : public Resources
{
public:
  struct FrameBuffer
  {
    int  renderWidth  = 0;
    int  renderHeight = 0;
    int  supersample  = 0;
    bool useResolved  = false;
    bool vsync        = false;
    int  msaa         = 0;

    VkViewport viewport;
    VkViewport viewportUI;
    VkRect2D   scissor;
    VkRect2D   scissorUI;

    VkRenderPass passClear    = VK_NULL_HANDLE;
    VkRenderPass passPreserve = VK_NULL_HANDLE;
    VkRenderPass passUI       = VK_NULL_HANDLE;

    VkFramebuffer fboScene = VK_NULL_HANDLE;
    VkFramebuffer fboUI    = VK_NULL_HANDLE;

    VkImage imgColor         = VK_NULL_HANDLE;
    VkImage imgColorResolved = VK_NULL_HANDLE;
    VkImage imgDepthStencil  = VK_NULL_HANDLE;

    VkImageView viewColor         = VK_NULL_HANDLE;
    VkImageView viewColorResolved = VK_NULL_HANDLE;
    VkImageView viewDepthStencil  = VK_NULL_HANDLE;

    nvvk::DeviceMemoryAllocator memAllocator;
  };

  struct Common
  {
    nvvk::AllocationID     viewAID;
    VkBuffer               viewBuffer;
    VkDescriptorBufferInfo viewInfo;

    nvvk::AllocationID     statsAID;
    VkBuffer               statsBuffer;
    VkDescriptorBufferInfo statsInfo;

    nvvk::AllocationID     statsReadAID;
    VkBuffer               statsReadBuffer;
    VkDescriptorBufferInfo statsReadInfo;
  };

  struct ShaderModuleIDs
  {
    nvvk::ShaderModuleID object_vertex;
    nvvk::ShaderModuleID object_fragment;
    nvvk::ShaderModuleID object_mesh;
    nvvk::ShaderModuleID object_task_mesh;
    nvvk::ShaderModuleID object_task;
    nvvk::ShaderModuleID bbox_vertex;
    nvvk::ShaderModuleID bbox_geometry;
    nvvk::ShaderModuleID bbox_fragment;
  };

  enum DrawMode
  {
    MODE_REGULAR,
    MODE_BBOX,
    MODE_MESH,
    MODE_TASK_MESH,
    NUM_MODES,
  };

  struct DrawSetup
  {
    VkPipeline                                pipeline       = VK_NULL_HANDLE;
    VkPipeline                                pipelineNoTask = VK_NULL_HANDLE;
    nvvk::TDescriptorSetContainer<DSET_COUNT> container;
  };


  bool m_withinFrame       = false;
  bool m_nativeMeshSupport = false;

  nvvk::ShaderModuleManager m_shaderManager;
  ShaderModuleIDs           m_shaders;

#if HAS_OPENGL
  //nvvk::Context m_ctxContent;
  VkSemaphore   m_semImageWritten;
  VkSemaphore   m_semImageRead;
  nvvk::Context m_contextInstance;
#else
  const nvvk::SwapChain* m_swapChain;
#endif
  nvvk::Context* m_context = nullptr;

  VkDevice                     m_device    = VK_NULL_HANDLE;
  VkPhysicalDevice             m_physical;
  VkQueue                      m_queue;
  uint32_t                     m_queueFamily;

  nvvk::DeviceMemoryAllocator m_memAllocator;
  nvvk::RingFences            m_ringFences;
  nvvk::RingCmdPool           m_ringCmdPool;


  nvvk::BatchSubmission m_submission;
  bool                  m_submissionWaitForRead;

  FrameBuffer m_framebuffer;
  Common      m_common;
  CadSceneVK  m_scene;

  DrawSetup m_setupRegular;
  DrawSetup m_setupBbox;
  DrawSetup m_setupMeshTask;

  nvvk::ProfilerVK m_profilerVK;

  size_t m_pipeChangeID;
  size_t m_fboChangeID;

  ResourcesVK() {}

  static ResourcesVK* get()
  {
    static ResourcesVK res;

    return &res;
  }
  static bool isAvailable();

#if HAS_OPENGL
  bool init(nvgl::ContextWindow* window, nvh::Profiler* profiler);
#else
  bool                   init(nvvk::Context* context, nvvk::SwapChain* swapChain, nvh::Profiler* profiler);
#endif
  void deinit() override;

  void initPipes();
  void deinitPipes();
  bool hasPipes() { return m_setupRegular.pipeline != 0; }

  bool initPrograms(const std::string& path, const std::string& prepend) override;
  void reloadPrograms(const std::string& prepend) override;

  void updatedPrograms();
  void deinitPrograms();

  bool initFramebuffer(int width, int height, int supersample, bool vsync) override;
  void deinitFramebuffer();

  bool initScene(const CadScene&) override;
  void deinitScene() override;

  void synchronize();

  void beginFrame() override;
  void blitFrame(const FrameConfig& global) override;
  void endFrame() override;

  void cmdCopyStats(VkCommandBuffer cmd) const;
  void getStats(CullStats& stats) override;

  nvmath::mat4f perspectiveProjection(float fovy, float aspect, float nearPlane, float farPlane) const override;

  //////////////////////////////////////////////////////////////////////////

  VkRenderPass createPass(bool clear, int msaa);
  VkRenderPass createPassUI(int msaa);

  VkCommandBuffer createCmdBuffer(VkCommandPool pool, bool singleshot, bool primary, bool secondaryInClear) const;
  VkCommandBuffer createTempCmdBuffer(bool primary = true, bool secondaryInClear = false);

  VkCommandBuffer createBoundingBoxCmdBuffer(VkCommandPool pool, const class RenderList* NV_RESTRICT list) const;

  // submit for batched execution
  void submissionEnqueue(VkCommandBuffer cmdbuffer) { m_submission.enqueue(cmdbuffer); }
  void submissionEnqueue(uint32_t num, const VkCommandBuffer* cmdbuffers) { m_submission.enqueue(num, cmdbuffers); }
  // perform queue submit
  void submissionExecute(VkFence fence = NULL, bool useImageReadWait = false, bool useImageWriteSignals = false);

  // synchronizes to queue
  void resetTempResources();

  void cmdBeginRenderPass(VkCommandBuffer cmd, bool clear, bool hasSecondary = false) const;
  void cmdPipelineBarrier(VkCommandBuffer cmd) const;
  void cmdDynamicState(VkCommandBuffer cmd) const;
  void cmdImageTransition(VkCommandBuffer    cmd,
                          VkImage            img,
                          VkImageAspectFlags aspects,
                          VkAccessFlags      src,
                          VkAccessFlags      dst,
                          VkImageLayout      oldLayout,
                          VkImageLayout      newLayout) const;
  void cmdBegin(VkCommandBuffer cmd, bool singleshot, bool primary, bool secondaryInClear) const;
};

}  // namespace meshlettest
