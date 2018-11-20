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

#include "resources.hpp"
#include "vulkan/vulkan.h"
#include "cadscene_vk.hpp"

#include <nv_helpers/tnulled.hpp>
#include <nv_helpers_vk/base_vk.hpp>
#include <nv_helpers_vk/swapchain_vk.hpp>
#include <nv_helpers_vk/shadermodulemanager_vk.hpp>
#include <nv_helpers_vk/window_vk.hpp>
#include <nv_helpers_vk/extensions_vk.hpp>

class NVPWindow;

#define DSET_COUNT  3

namespace meshlettest {  
  template <typename T>
  using TNulled = nv_helpers::TNulled<T>;
 

  class ResourcesVK : public Resources, 
    public nv_helpers::Profiler::GPUInterface, 
    public nv_helpers_vk::DeviceUtils, 
    public nv_helpers_vk::TempSubmissionInterface
  {
  public:

    struct FrameBuffer {
      int     renderWidth = 0;
      int     renderHeight = 0;
      int     supersample = 0;
      bool    useResolved = false;
      bool    vsync = false;
      int     msaa = 0;

      VkViewport    viewport;
      VkViewport    viewportUI;
      VkRect2D      scissor;
      VkRect2D      scissorUI;

      VkRenderPass  passClear = VK_NULL_HANDLE;
      VkRenderPass  passPreserve = VK_NULL_HANDLE;
      VkRenderPass  passUI = VK_NULL_HANDLE;

      VkFramebuffer fboScene = VK_NULL_HANDLE;
      VkFramebuffer fboUI = VK_NULL_HANDLE;
      
      VkImage imgColor = VK_NULL_HANDLE;
      VkImage imgColorResolved = VK_NULL_HANDLE;
      VkImage imgDepthStencil = VK_NULL_HANDLE;

      VkImageView  viewColor = VK_NULL_HANDLE;
      VkImageView  viewColorResolved = VK_NULL_HANDLE;
      VkImageView  viewDepthStencil = VK_NULL_HANDLE;

      nv_helpers_vk::BasicDeviceMemoryAllocator  memAllocator;
    };

    struct Common {
      nv_helpers_vk::AllocationID viewAID;
      VkBuffer                    viewBuffer;
      VkDescriptorBufferInfo      viewInfo;

      nv_helpers_vk::AllocationID statsAID;
      VkBuffer                    statsBuffer;
      VkDescriptorBufferInfo      statsInfo;

      nv_helpers_vk::AllocationID statsReadAID;
      VkBuffer                    statsReadBuffer;
      VkDescriptorBufferInfo      statsReadInfo;
    };
    
    struct ShaderModuleIDs {
      nv_helpers_vk::ShaderModuleManager::ShaderModuleID
        object_vertex,
        object_fragment,

        object_mesh,
        object_task_mesh,
        object_task,

        bbox_vertex,
        bbox_geometry,
        bbox_fragment;
    };

    enum DrawMode {
      MODE_REGULAR,
      MODE_BBOX,
      MODE_MESH,
      MODE_TASK_MESH,
      NUM_MODES,
    };

    struct DrawSetup
    {
      TNulled<VkPipeline>                                     pipeline;
      TNulled<VkPipeline>                                     pipelineNoTask;
      nv_helpers_vk::DescriptorPipelineContainer<DSET_COUNT>  container;
    };


    bool          m_withinFrame = false;
    bool          m_nativeMeshSupport = false;

    nv_helpers_vk::ShaderModuleManager            m_shaderManager;
    ShaderModuleIDs                               m_shaders;

  #if HAS_OPENGL
    nv_helpers_vk::InstanceDeviceContext          m_ctxContent;
    VkSemaphore                                   m_semImageWritten;
    VkSemaphore                                   m_semImageRead;
  #else
    const nv_helpers_vk::SwapChain*               m_swapChain;
  #endif
    const nv_helpers_vk::InstanceDeviceContext*   m_ctx;
    const nv_helpers_vk::PhysicalInfo*            m_physical;
    VkQueue                                       m_queue;
    uint32_t                                      m_queueFamily;

    nv_helpers_vk::BasicDeviceMemoryAllocator     m_memAllocator;

    FrameBuffer                                   m_framebuffer;

    nv_helpers_vk::RingFences                     m_ringFences;
    nv_helpers_vk::RingCmdPool                    m_ringCmdPool;

    bool                                          m_submissionWaitForRead;
    nv_helpers_vk::Submission                     m_submission;

    Common                        m_common;
    CadSceneVK                    m_scene;

    DrawSetup                     m_setupRegular;
    DrawSetup                     m_setupBbox;
    DrawSetup                     m_setupMeshTask;

    TNulled<VkQueryPool>      m_timePool;
    double                    m_timeStampFrequency;
    VkBool32                  m_timeStampsSupported;

    size_t                    m_pipeIncarnation;
    size_t                    m_fboIncarnation;

    ResourcesVK() {}

    static ResourcesVK* get() {
      static ResourcesVK res;

      return &res;
    }
    static bool ResourcesVK::isAvailable();

    bool init(NVPWindow *window);
    void deinit();

    void initPipes();
    void deinitPipes();
    bool hasPipes(){
      return m_setupRegular.pipeline != 0;
    }

    bool initPrograms(const std::string& path, const std::string& prepend) override;
    void reloadPrograms(const std::string& prepend) override;

    void updatedPrograms();
    void deinitPrograms();

    bool initFramebuffer(int width, int height, int supersample, bool vsync) override;
    void deinitFramebuffer();

    bool initScene(const CadScene&) override;
    void deinitScene() override;

    void synchronize();
    void initTimers(unsigned int n);
    void deinitTimers();

    void beginFrame() override;
    void blitFrame(const FrameConfig& global) override;
    void endFrame() override;

    void cmdCopyStats(VkCommandBuffer cmd) const;
    void getStats(CullStats& stats) override;

    nv_math::mat4f perspectiveProjection(float fovy, float aspect, float nearPlane, float farPlane) const override;

    nv_helpers::Profiler::GPUInterface*  getTimerInterface() override;

    const char* TimerTypeName() override;
    bool    TimerAvailable(nv_helpers::Profiler::TimerIdx idx) override;
    void    TimerSetup(nv_helpers::Profiler::TimerIdx idx) override;
    void    TimerSetup(nv_helpers::Profiler::TimerIdx idx, void* payload) override;
    unsigned long long  TimerResult(nv_helpers::Profiler::TimerIdx idxBegin, nv_helpers::Profiler::TimerIdx idxEnd) override;
    void    TimerEnsureSize(unsigned int slots) override;
    void    TimerFlush() override;


    VkCommandBuffer tempSubmissionCreateCommandBuffer(bool primary, VkQueueFlags preferredQueue = 0) override;
    void            tempSubmissionEnqueue(VkCommandBuffer cmd, VkQueueFlags preferredQueue = 0) override;
    void            tempSubmissionSubmit(bool sync, VkFence fence = 0, VkQueueFlags preferredQueue = 0, uint32_t deviceMask = 0) override;

    //////////////////////////////////////////////////////////////////////////

    VkRenderPass    createPass(bool clear, int msaa);
    VkRenderPass    createPassUI(int msaa);

    void            fillBuffer(nv_helpers_vk::BasicStagingBuffer& staging, VkBuffer buffer, size_t offset, size_t size, const void* data);
    VkBuffer        createBuffer(size_t size, VkFlags usage, VkDeviceMemory &bufferMem, VkFlags memProps = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VkBuffer        createAndFillBuffer(nv_helpers_vk::BasicStagingBuffer& staging, size_t size, const void* data, VkFlags usage, VkDeviceMemory &bufferMem,
                                        VkFlags memProps = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    VkCommandBuffer   createCmdBuffer(VkCommandPool pool, bool singleshot, bool primary, bool secondaryInClear) const;
    VkCommandBuffer   createTempCmdBuffer(bool primary=true, bool secondaryInClear=false);

    VkCommandBuffer   createBoundingBoxCmdBuffer(VkCommandPool pool, const class RenderList* NV_RESTRICT list) const;


    VkResult allocMemAndBindBuffer(VkBuffer obj, VkDeviceMemory &gpuMem, VkFlags memProps = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
    {
      return DeviceUtils::allocMemAndBindBuffer(obj, m_physical->memoryProperties, gpuMem, memProps);
    }

    // submit for batched execution
    void      submissionEnqueue(VkCommandBuffer cmdbuffer)
    {
      m_submission.enqueue(cmdbuffer);
    }
    void      submissionEnqueue(uint32_t num, const VkCommandBuffer* cmdbuffers)
    {
      m_submission.enqueue(num, cmdbuffers);
    }
    // perform queue submit
    void      submissionExecute(VkFence fence=NULL, bool useImageReadWait=false, bool useImageWriteSignals=false);

    // synchronizes to queue
    void          resetTempResources(); 

    void          cmdBeginRenderPass(VkCommandBuffer cmd, bool clear, bool hasSecondary=false) const;
    void          cmdPipelineBarrier(VkCommandBuffer cmd) const;
    void          cmdDynamicState   (VkCommandBuffer cmd) const;
    void          cmdImageTransition(VkCommandBuffer cmd, VkImage img, VkImageAspectFlags aspects, VkAccessFlags src, VkAccessFlags dst, VkImageLayout oldLayout, VkImageLayout newLayout) const;
    void          cmdBegin          (VkCommandBuffer cmd, bool singleshot, bool primary, bool secondaryInClear) const;
  };

}

