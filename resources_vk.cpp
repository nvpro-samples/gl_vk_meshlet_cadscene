/*-----------------------------------------------------------------------
  Copyright (c) 2014-2016, NVIDIA. All rights reserved.
  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions
  are met:
   * Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.
   * Neither the name of its contributors may be used to endorse 
     or promote products derived from this software without specific
     prior written permission.
  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
  PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
  OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
-----------------------------------------------------------------------*/
/* Contact ckubisch@nvidia.com (Christoph Kubisch) for feedback */

#include "resources_vk.hpp"
#include "renderer.hpp"

#include <imgui/imgui_impl_vk.h>

#include <algorithm>

#if HAS_OPENGL
#include <nvgl/extensions_gl.hpp>
#endif
#include <nvh/nvprint.hpp>

#include "nvmeshlet_builder.hpp"

extern bool vulkanInitLibrary();

namespace meshlettest {

extern void setupVulkanContextInfo(nvvk::ContextCreateInfo& info);


bool ResourcesVK::isAvailable()
{
  static bool result = false;
  static bool s_init = false;

  if(s_init)
  {
    return result;
  }

  s_init = true;
  result = vulkanInitLibrary();

  return result;
}

/////////////////////////////////////////////////////////////////////////////////


void ResourcesVK::submissionExecute(VkFence fence, bool useImageReadWait, bool useImageWriteSignals)
{
  if(useImageReadWait && m_submissionWaitForRead)
  {
#if HAS_OPENGL
    VkSemaphore semRead = m_semImageRead;
#else
    VkSemaphore semRead    = m_swapChain->getActiveReadSemaphore();
#endif
    if(semRead)
    {
      m_submission.enqueueWait(semRead, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
    }
    m_submissionWaitForRead = false;
  }

  if(useImageWriteSignals)
  {
#if HAS_OPENGL
    VkSemaphore semWritten = m_semImageWritten;
#else
    VkSemaphore semWritten = m_swapChain->getActiveWrittenSemaphore();
#endif
    if(semWritten)
    {
      m_submission.enqueueSignal(semWritten);
    }
  }

  m_submission.execute(fence);
}

void ResourcesVK::beginFrame()
{
  assert(!m_withinFrame);
  m_withinFrame           = true;
  m_submissionWaitForRead = true;
  m_ringFences.setCycleAndWait(m_frame);
  m_ringCmdPool.setCycle(m_frame);
}

void ResourcesVK::endFrame()
{
  submissionExecute(m_ringFences.getFence(), true, true);
  assert(m_withinFrame);
  m_withinFrame = false;
#if HAS_OPENGL
  {
    // blit to gl backbuffer
    glDisable(GL_DEPTH_TEST);
    glViewport(0, 0, m_framebuffer.renderWidth / m_framebuffer.supersample, m_framebuffer.renderHeight / m_framebuffer.supersample);
    glWaitVkSemaphoreNV((GLuint64)m_semImageWritten);
    glDrawVkImageNV((GLuint64)(VkImage)(m_framebuffer.useResolved ? m_framebuffer.imgColorResolved : m_framebuffer.imgColor),
                    0, 0, 0, m_framebuffer.renderWidth / m_framebuffer.supersample,
                    m_framebuffer.renderHeight / m_framebuffer.supersample, 0, 0, 1, 1, 0);
    glEnable(GL_DEPTH_TEST);
    glSignalVkSemaphoreNV((GLuint64)m_semImageRead);
  }
#endif
}

void ResourcesVK::blitFrame(const FrameConfig& global)
{
  VkCommandBuffer cmd = createTempCmdBuffer();

  nvh::Profiler::SectionID sec = m_profilerVK.beginSection("BltUI", cmd);

  VkImage imageBlitRead = m_framebuffer.imgColor;

  if(m_framebuffer.useResolved)
  {
    cmdImageTransition(cmd, m_framebuffer.imgColor, VK_IMAGE_ASPECT_COLOR_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                       VK_ACCESS_TRANSFER_READ_BIT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

    // blit to resolved
    VkImageBlit region               = {0};
    region.dstOffsets[1].x           = global.winWidth;
    region.dstOffsets[1].y           = global.winHeight;
    region.dstOffsets[1].z           = 1;
    region.srcOffsets[1].x           = m_framebuffer.renderWidth;
    region.srcOffsets[1].y           = m_framebuffer.renderHeight;
    region.srcOffsets[1].z           = 1;
    region.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.dstSubresource.layerCount = 1;
    region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.srcSubresource.layerCount = 1;

    imageBlitRead = m_framebuffer.imgColorResolved;

    vkCmdBlitImage(cmd, m_framebuffer.imgColor, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, imageBlitRead,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region, VK_FILTER_LINEAR);
  }

  // It would be better to render the ui ontop of backbuffer
  // instead of using the "resolved" image here, as it would avoid an additional
  // blit. However, for the simplicity to pass a final image in the OpenGL mode
  // we avoid rendering to backbuffer directly.

  if(global.imguiDrawData)
  {
    VkRenderPassBeginInfo renderPassBeginInfo    = {VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    renderPassBeginInfo.renderPass               = m_framebuffer.passUI;
    renderPassBeginInfo.framebuffer              = m_framebuffer.fboUI;
    renderPassBeginInfo.renderArea.offset.x      = 0;
    renderPassBeginInfo.renderArea.offset.y      = 0;
    renderPassBeginInfo.renderArea.extent.width  = global.winWidth;
    renderPassBeginInfo.renderArea.extent.height = global.winHeight;
    renderPassBeginInfo.clearValueCount          = 0;
    renderPassBeginInfo.pClearValues             = nullptr;

    vkCmdBeginRenderPass(cmd, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdSetViewport(cmd, 0, 1, &m_framebuffer.viewportUI);
    vkCmdSetScissor(cmd, 0, 1, &m_framebuffer.scissorUI);

    ImGui::RenderDrawDataVK(cmd, global.imguiDrawData);

    vkCmdEndRenderPass(cmd);

    // turns imageBlitRead to VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
  }
  else
  {
    if(m_framebuffer.useResolved)
    {
      cmdImageTransition(cmd, m_framebuffer.imgColorResolved, VK_IMAGE_ASPECT_COLOR_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                         VK_ACCESS_TRANSFER_READ_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    }
    else
    {
      cmdImageTransition(cmd, m_framebuffer.imgColor, VK_IMAGE_ASPECT_COLOR_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                         VK_ACCESS_TRANSFER_READ_BIT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    }
  }


#if !HAS_OPENGL
  {
    // blit to vk backbuffer
    VkImageBlit region               = {0};
    region.dstOffsets[1].x           = global.winWidth;
    region.dstOffsets[1].y           = global.winHeight;
    region.dstOffsets[1].z           = 1;
    region.srcOffsets[1].x           = global.winWidth;
    region.srcOffsets[1].y           = global.winHeight;
    region.srcOffsets[1].z           = 1;
    region.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.dstSubresource.layerCount = 1;
    region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.srcSubresource.layerCount = 1;

    cmdImageTransition(cmd, m_swapChain->getActiveImage(), VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
                       VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    vkCmdBlitImage(cmd, imageBlitRead, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, m_swapChain->getActiveImage(),
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region, VK_FILTER_NEAREST);

    cmdImageTransition(cmd, m_swapChain->getActiveImage(), VK_IMAGE_ASPECT_COLOR_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, 0,
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
  }
#endif

  if(m_framebuffer.useResolved)
  {
    cmdImageTransition(cmd, m_framebuffer.imgColorResolved, VK_IMAGE_ASPECT_COLOR_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                       VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
  }

  m_profilerVK.endSection(sec, cmd);

  vkEndCommandBuffer(cmd);
  submissionEnqueue(cmd);
}

void ResourcesVK::cmdCopyStats(VkCommandBuffer cmd) const
{
  VkBufferCopy region;
  region.size      = sizeof(CullStats);
  region.srcOffset = 0;
  region.dstOffset = m_ringFences.getCycleIndex() * sizeof(CullStats);
  vkCmdCopyBuffer(cmd, m_common.statsBuffer, m_common.statsReadBuffer, 1, &region);
}

void ResourcesVK::getStats(CullStats& stats)
{
  const CullStats* pStats = (const CullStats*)m_memAllocator.map(m_common.statsReadAID);
  stats                   = pStats[m_ringFences.getCycleIndex()];
  m_memAllocator.unmap(m_common.statsReadAID);
}

#if HAS_OPENGL
bool ResourcesVK::init(nvgl::ContextWindow* contextWindow, nvh::Profiler* profiler)
#else
bool ResourcesVK::init(nvvk::Context* context, nvvk::SwapChain* swapChain, nvh::Profiler* profiler)
#endif
{
  m_fboChangeID  = 0;
  m_pipeChangeID = 0;

#if HAS_OPENGL
  {
    nvvk::ContextCreateInfo info;
    info.compatibleDeviceIndex = Resources::s_vkDevice;
    setupVulkanContextInfo(info);
    m_context = &m_contextInstance;
    if(!m_context->init(info))
    {
      LOGI("vulkan device create failed (use debug build for more information)\n");
      exit(-1);
      return false;
    }
  }

  {
    // OpenGL drawing
    VkSemaphoreCreateInfo semCreateInfo = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    vkCreateSemaphore(m_context->m_device, &semCreateInfo, nullptr, &m_semImageRead);
    vkCreateSemaphore(m_context->m_device, &semCreateInfo, nullptr, &m_semImageWritten);

    // fire read to ensure queuesubmit never waits
    glSignalVkSemaphoreNV((GLuint64)m_semImageRead);
    glFlush();
  }
#else
  {
    m_context = context;
    m_swapChain = swapChain;
  }
#endif

  m_device      = m_context->m_device;
  m_physical    = m_context->m_physicalDevice;
  m_queue       = m_context->m_queueGCT.queue;
  m_queueFamily = m_context->m_queueGCT.familyIndex;

  LOGI("Vk device: %s\n", m_context->m_physicalInfo.properties10.deviceName);

  initAlignedSizes((uint32_t)m_context->m_physicalInfo.properties10.limits.minUniformBufferOffsetAlignment);

  m_nativeMeshSupport = m_context->hasDeviceExtension(VK_NV_MESH_SHADER_EXTENSION_NAME);

  m_profilerVK = nvvk::ProfilerVK(profiler);
  m_profilerVK.init(m_device, m_physical);

  // submission queue
  m_submission.init(m_queue);

  // fences
  m_ringFences.init(m_device);

  // temp cmd pool
  m_ringCmdPool.init(m_device, m_queueFamily, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT);

  // Create the render passes
  {
    m_framebuffer.passClear    = createPass(true, m_framebuffer.msaa);
    m_framebuffer.passPreserve = createPass(false, m_framebuffer.msaa);
    m_framebuffer.passUI       = createPassUI(m_framebuffer.msaa);
  }

  // device mem allocator
  m_memAllocator.init(m_device, m_physical);

  {
    // common

    m_common.viewBuffer = m_memAllocator.createBuffer(sizeof(SceneData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, m_common.viewAID);
    m_common.viewInfo = {m_common.viewBuffer, 0, sizeof(SceneData)};

    m_common.statsBuffer = m_memAllocator.createBuffer(
        sizeof(CullStats), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, m_common.statsAID);
    m_common.statsInfo = {m_common.statsBuffer, 0, sizeof(CullStats)};

    m_common.statsReadBuffer =
        m_memAllocator.createBuffer(sizeof(CullStats) * nvvk::DEFAULT_RING_SIZE, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                    m_common.statsReadAID, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
    m_common.statsReadInfo = {m_common.statsReadBuffer, 0, sizeof(CullStats) * nvvk::DEFAULT_RING_SIZE};
  }

  {

    ///////////////////////////////////////////////////////////////////////////////////////////
    {
      // REGULAR
      DrawSetup& setup = m_setupRegular;
      setup.container.init(m_device);

      auto& bindingsScene = setup.container.at(DSET_SCENE);
      // UBO SCENE
      bindingsScene.addBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0);
      bindingsScene.initLayout();
      // UBO OBJECT
      auto& bindingsObject = setup.container.at(DSET_OBJECT);
      bindingsObject.addBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1,
                                VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0);
      bindingsObject.initLayout();

      setup.container.initPipeLayout(0, 2, uint32_t(0));
    }

    {
      // BBOX
      DrawSetup& setup = m_setupBbox;
      setup.container.init(m_device);
      // UBO SCENE
      auto& bindingsScene = setup.container.at(DSET_SCENE);
      bindingsScene.addBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_GEOMETRY_BIT, 0);
      bindingsScene.initLayout();
      // UBO OBJECT
      auto& bindingsObject = setup.container.at(DSET_OBJECT);
      bindingsObject.addBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1,
                                VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_GEOMETRY_BIT, 0);
      bindingsObject.initLayout();
      // UBO GEOMETRY
      auto& bindingsGeometry = setup.container.at(DSET_GEOMETRY);
      bindingsGeometry.addBinding(GEOMETRY_SSBO_MESHLETDESC, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, 0);
      bindingsGeometry.addBinding(GEOMETRY_SSBO_PRIM, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, 0);
      bindingsGeometry.addBinding(GEOMETRY_TEX_IBO, VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, 0);
      bindingsGeometry.addBinding(GEOMETRY_TEX_VBO, VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, 0);
      bindingsGeometry.addBinding(GEOMETRY_TEX_ABO, VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, 0);

      bindingsGeometry.initLayout();

#if USE_PER_GEOMETRY_VIEWS
      setup.container.initPipeLayout(0, 3);
#else
      VkPushConstantRange range;
      range.offset = 0;
      range.size = sizeof(uint32_t) * 4;
      range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
      setup.container.initPipeLayout(0, 3, 1, &range);
#endif
    }

    if(m_nativeMeshSupport)
    {
      VkPipelineStageFlags stageMesh = VK_SHADER_STAGE_MESH_BIT_NV;
      VkPipelineStageFlags stageTask = VK_SHADER_STAGE_TASK_BIT_NV;

      {
        // TASK
        DrawSetup& setup = m_setupMeshTask;
        setup.container.init(m_device);
        // UBO SCENE
        auto& bindingsScene = setup.container.at(DSET_SCENE);
        bindingsScene.addBinding(SCENE_UBO_VIEW, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
                                 stageTask | stageMesh | VK_SHADER_STAGE_FRAGMENT_BIT, 0);
        bindingsScene.addBinding(SCENE_SSBO_STATS, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, stageTask | stageMesh, 0);
        bindingsScene.initLayout();
        // UBO OBJECT
        auto& bindingsObject = setup.container.at(DSET_OBJECT);
        bindingsObject.addBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1,
                                  stageTask | stageMesh | VK_SHADER_STAGE_FRAGMENT_BIT, 0);
        bindingsObject.initLayout();
        // UBO GEOMETRY
        auto& bindingsGeometry = setup.container.at(DSET_GEOMETRY);
        bindingsGeometry.addBinding(GEOMETRY_SSBO_MESHLETDESC, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, stageTask | stageMesh, 0);
        bindingsGeometry.addBinding(GEOMETRY_SSBO_PRIM, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, stageMesh, 0);
        bindingsGeometry.addBinding(GEOMETRY_TEX_IBO, VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1, stageMesh, 0);
        bindingsGeometry.addBinding(GEOMETRY_TEX_VBO, VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1, stageMesh, 0);
        bindingsGeometry.addBinding(GEOMETRY_TEX_ABO, VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1, stageMesh, 0);
        bindingsGeometry.initLayout();

        VkPushConstantRange ranges[2];
#if USE_PER_GEOMETRY_VIEWS
        ranges[0].offset     = (USE_PER_GEOMETRY_VIEWS ? 0 : sizeof(uint32_t) * 4);
        ranges[0].size       = sizeof(uint32_t) * 4;
        ranges[0].stageFlags = VK_SHADER_STAGE_TASK_BIT_NV;
#else
        ranges[0].offset = 0;
        ranges[0].size = sizeof(uint32_t) * 8;
        ranges[0].stageFlags = VK_SHADER_STAGE_TASK_BIT_NV;
        ranges[1].offset = 0;
        ranges[1].size = sizeof(uint32_t) * 4;
        ranges[1].stageFlags = VK_SHADER_STAGE_MESH_BIT_NV;
#endif
        setup.container.initPipeLayout(0, 3, USE_PER_GEOMETRY_VIEWS ? 1 : 2, ranges);
      }
    }
  }

  {
    ImGui::InitVK(m_context->m_device, m_context->m_physicalDevice, m_context->m_queueGCT,
                  m_context->m_queueGCT.familyIndex, m_framebuffer.passUI);
  }

  return true;
}

void ResourcesVK::deinit()
{
  synchronize();

  ImGui::ShutdownVK();

  {
    vkDestroyBuffer(m_device, m_common.viewBuffer, NULL);
    m_memAllocator.free(m_common.viewAID);
    vkDestroyBuffer(m_device, m_common.statsBuffer, NULL);
    m_memAllocator.free(m_common.statsAID);
    vkDestroyBuffer(m_device, m_common.statsReadBuffer, NULL);
    m_memAllocator.free(m_common.statsReadAID);
  }


  m_ringFences.deinit();
  m_ringCmdPool.deinit();

  deinitScene();
  deinitFramebuffer();
  deinitPipes();
  deinitPrograms();

  m_profilerVK.deinit();

  vkDestroyRenderPass(m_device, m_framebuffer.passClear, NULL);
  vkDestroyRenderPass(m_device, m_framebuffer.passPreserve, NULL);
  vkDestroyRenderPass(m_device, m_framebuffer.passUI, NULL);

  m_setupRegular.container.deinitLayouts();
  m_setupBbox.container.deinitLayouts();

  if(m_nativeMeshSupport)
  {
    m_setupMeshTask.container.deinitLayouts();
  }

  m_memAllocator.deinit();

#if HAS_OPENGL
  vkDestroySemaphore(m_device, m_semImageRead, NULL);
  vkDestroySemaphore(m_device, m_semImageWritten, NULL);
  m_device = NULL;

  m_context->deinit();
#endif
}

bool ResourcesVK::initPrograms(const std::string& path, const std::string& prepend)
{
  m_shaderManager.init(m_device);
  m_shaderManager.m_filetype       = nvh::ShaderFileManager::FILETYPE_GLSL;

  m_shaderManager.addDirectory(path);
  m_shaderManager.addDirectory(std::string("GLSL_" PROJECT_NAME));
  m_shaderManager.addDirectory(path + std::string(PROJECT_RELDIRECTORY));

  m_shaderManager.registerInclude("draw.frag.glsl");
  m_shaderManager.registerInclude("nvmeshlet_utils.glsl");
  m_shaderManager.registerInclude("config.h");
  m_shaderManager.registerInclude("common.h");

  m_shaderManager.m_prepend = std::string("#define IS_VULKAN 1\n") + prepend;

  ///////////////////////////////////////////////////////////////////////////////////////////
  {
    m_shaders.object_vertex   = m_shaderManager.createShaderModule(VK_SHADER_STAGE_VERTEX_BIT, "draw.vert.glsl");
    m_shaders.object_fragment = m_shaderManager.createShaderModule(VK_SHADER_STAGE_FRAGMENT_BIT, "draw.frag.glsl");
  }

  {
    m_shaders.bbox_vertex   = m_shaderManager.createShaderModule(VK_SHADER_STAGE_VERTEX_BIT, "meshletbbox.vert.glsl");
    m_shaders.bbox_geometry = m_shaderManager.createShaderModule(VK_SHADER_STAGE_GEOMETRY_BIT, "meshletbbox.geo.glsl");
    m_shaders.bbox_fragment = m_shaderManager.createShaderModule(VK_SHADER_STAGE_FRAGMENT_BIT, "meshletbbox.frag.glsl");
  }

  if(m_nativeMeshSupport)
  {
    m_shaders.object_mesh = m_shaderManager.createShaderModule(VK_SHADER_STAGE_MESH_BIT_NV, "drawmeshlet.mesh.glsl",
                                                               "#define USE_TASK_STAGE 0\n");
    m_shaders.object_task_mesh = m_shaderManager.createShaderModule(VK_SHADER_STAGE_MESH_BIT_NV, "drawmeshlet.mesh.glsl",
                                                                    "#define USE_TASK_STAGE 1\n");
    m_shaders.object_task = m_shaderManager.createShaderModule(VK_SHADER_STAGE_TASK_BIT_NV, "drawmeshlet.task.glsl",
                                                               "#define USE_TASK_STAGE 1\n");
  }
  ///////////////////////////////////////////////////////////////////////////////////////////

  bool valid = m_shaderManager.areShaderModulesValid();

  if(valid)
  {
    updatedPrograms();
  }

  return valid;
}

void ResourcesVK::reloadPrograms(const std::string& prepend)
{
  m_shaderManager.m_prepend = std::string("#define IS_VULKAN 1\n") + prepend;
  m_shaderManager.reloadShaderModules();
  updatedPrograms();
}

void ResourcesVK::updatedPrograms()
{
  initPipes();
}

void ResourcesVK::deinitPrograms()
{
  m_shaderManager.deinit();
}

static VkSampleCountFlagBits getSampleCountFlagBits(int msaa)
{
  switch(msaa)
  {
    case 2:
      return VK_SAMPLE_COUNT_2_BIT;
    case 4:
      return VK_SAMPLE_COUNT_4_BIT;
    case 8:
      return VK_SAMPLE_COUNT_8_BIT;
    default:
      return VK_SAMPLE_COUNT_1_BIT;
  }
}

VkRenderPass ResourcesVK::createPass(bool clear, int msaa)
{
  VkResult result;

  VkAttachmentLoadOp loadOp = clear ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;

  VkSampleCountFlagBits samplesUsed = getSampleCountFlagBits(msaa);

  // Create the render pass
  VkAttachmentDescription attachments[2] = {};
  attachments[0].format                  = VK_FORMAT_R8G8B8A8_UNORM;
  attachments[0].samples                 = samplesUsed;
  attachments[0].loadOp                  = loadOp;
  attachments[0].storeOp                 = VK_ATTACHMENT_STORE_OP_STORE;
  attachments[0].initialLayout           = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  attachments[0].finalLayout             = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  attachments[0].flags                   = 0;

  VkFormat depthStencilFormat = nvvk::findDepthStencilFormat(m_physical);

  attachments[1].format              = depthStencilFormat;
  attachments[1].samples             = samplesUsed;
  attachments[1].loadOp              = loadOp;
  attachments[1].storeOp             = VK_ATTACHMENT_STORE_OP_STORE;
  attachments[1].stencilLoadOp       = loadOp;
  attachments[1].stencilStoreOp      = VK_ATTACHMENT_STORE_OP_STORE;
  attachments[1].initialLayout       = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
  attachments[1].finalLayout         = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
  attachments[1].flags               = 0;
  VkSubpassDescription subpass       = {};
  subpass.pipelineBindPoint          = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.inputAttachmentCount       = 0;
  VkAttachmentReference colorRefs[1] = {{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}};
  subpass.colorAttachmentCount       = NV_ARRAY_SIZE(colorRefs);
  subpass.pColorAttachments          = colorRefs;
  VkAttachmentReference depthRefs[1] = {{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL}};
  subpass.pDepthStencilAttachment    = depthRefs;
  VkRenderPassCreateInfo rpInfo      = {VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
  rpInfo.attachmentCount             = NV_ARRAY_SIZE(attachments);
  rpInfo.pAttachments                = attachments;
  rpInfo.subpassCount                = 1;
  rpInfo.pSubpasses                  = &subpass;
  rpInfo.dependencyCount             = 0;

  VkRenderPass rp;
  result = vkCreateRenderPass(m_device, &rpInfo, NULL, &rp);
  assert(result == VK_SUCCESS);
  return rp;
}


VkRenderPass ResourcesVK::createPassUI(int msaa)
{
  // ui related
  // two cases:
  // if msaa we want to render into scene_color_resolved, which was DST_OPTIMAL
  // otherwise render into scene_color, which was VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
  VkImageLayout uiTargetLayout =
      m_framebuffer.useResolved ? VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  // Create the ui render pass
  VkAttachmentDescription attachments[1] = {};
  attachments[0].format                  = VK_FORMAT_R8G8B8A8_UNORM;
  attachments[0].samples                 = VK_SAMPLE_COUNT_1_BIT;
  attachments[0].loadOp                  = VK_ATTACHMENT_LOAD_OP_LOAD;
  attachments[0].storeOp                 = VK_ATTACHMENT_STORE_OP_STORE;
  attachments[0].initialLayout           = uiTargetLayout;
  attachments[0].finalLayout             = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;  // for blit operation
  attachments[0].flags                   = 0;

  VkSubpassDescription subpass       = {};
  subpass.pipelineBindPoint          = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.inputAttachmentCount       = 0;
  VkAttachmentReference colorRefs[1] = {{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}};
  subpass.colorAttachmentCount       = NV_ARRAY_SIZE(colorRefs);
  subpass.pColorAttachments          = colorRefs;
  subpass.pDepthStencilAttachment    = nullptr;
  VkRenderPassCreateInfo rpInfo      = {VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
  rpInfo.attachmentCount             = NV_ARRAY_SIZE(attachments);
  rpInfo.pAttachments                = attachments;
  rpInfo.subpassCount                = 1;
  rpInfo.pSubpasses                  = &subpass;
  rpInfo.dependencyCount             = 0;

  VkRenderPass rp;
  VkResult     result = vkCreateRenderPass(m_device, &rpInfo, NULL, &rp);
  assert(result == VK_SUCCESS);
  return rp;
}


bool ResourcesVK::initFramebuffer(int winWidth, int winHeight, int supersample, bool vsync)
{
  VkResult result;

  m_fboChangeID++;

  if(m_framebuffer.imgColor != 0)
  {
    deinitFramebuffer();
  }

  m_framebuffer.memAllocator.init(m_device, m_physical);

  int  oldMsaa     = m_framebuffer.msaa;
  bool oldResolved = m_framebuffer.supersample > 1;

  m_framebuffer.renderWidth  = winWidth * supersample;
  m_framebuffer.renderHeight = winHeight * supersample;
  m_framebuffer.supersample  = supersample;
  m_framebuffer.msaa         = 0;
  m_framebuffer.vsync        = vsync;

  LOGI("framebuffer: %d x %d (%d msaa)\n", m_framebuffer.renderWidth, m_framebuffer.renderHeight, m_framebuffer.msaa);

  m_framebuffer.useResolved = supersample > 1;

  if(oldMsaa != m_framebuffer.msaa || oldResolved != m_framebuffer.useResolved)
  {
    vkDestroyRenderPass(m_device, m_framebuffer.passClear, NULL);
    vkDestroyRenderPass(m_device, m_framebuffer.passPreserve, NULL);
    vkDestroyRenderPass(m_device, m_framebuffer.passUI, NULL);

    // recreate the render passes with new msaa setting
    m_framebuffer.passClear    = createPass(true, m_framebuffer.msaa);
    m_framebuffer.passPreserve = createPass(false, m_framebuffer.msaa);
    m_framebuffer.passUI       = createPassUI(m_framebuffer.msaa);
  }

  VkSampleCountFlagBits samplesUsed = getSampleCountFlagBits(m_framebuffer.msaa);

  // color
  VkImageCreateInfo cbImageInfo = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  cbImageInfo.imageType         = VK_IMAGE_TYPE_2D;
  cbImageInfo.format            = VK_FORMAT_R8G8B8A8_UNORM;
  cbImageInfo.extent.width      = m_framebuffer.renderWidth;
  cbImageInfo.extent.height     = m_framebuffer.renderHeight;
  cbImageInfo.extent.depth      = 1;
  cbImageInfo.mipLevels         = 1;
  cbImageInfo.arrayLayers       = 1;
  cbImageInfo.samples           = samplesUsed;
  cbImageInfo.tiling            = VK_IMAGE_TILING_OPTIMAL;
  cbImageInfo.usage             = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  cbImageInfo.flags             = 0;
  cbImageInfo.initialLayout     = VK_IMAGE_LAYOUT_UNDEFINED;

  {
    nvvk::AllocationID allocationId;
    m_framebuffer.imgColor = m_framebuffer.memAllocator.createImage(cbImageInfo, allocationId, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  }

  // depth stencil
  VkFormat depthStencilFormat = nvvk::findDepthStencilFormat(m_physical);

  VkImageCreateInfo dsImageInfo = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  dsImageInfo.imageType         = VK_IMAGE_TYPE_2D;
  dsImageInfo.format            = depthStencilFormat;
  dsImageInfo.extent.width      = m_framebuffer.renderWidth;
  dsImageInfo.extent.height     = m_framebuffer.renderHeight;
  dsImageInfo.extent.depth      = 1;
  dsImageInfo.mipLevels         = 1;
  dsImageInfo.arrayLayers       = 1;
  dsImageInfo.samples           = samplesUsed;
  dsImageInfo.tiling            = VK_IMAGE_TILING_OPTIMAL;
  dsImageInfo.usage             = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
  dsImageInfo.flags             = 0;
  dsImageInfo.initialLayout     = VK_IMAGE_LAYOUT_UNDEFINED;

  {
    nvvk::AllocationID allocationId;
    m_framebuffer.imgDepthStencil =
        m_framebuffer.memAllocator.createImage(dsImageInfo, allocationId, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  }

  if(m_framebuffer.useResolved)
  {
    // resolve image
    VkImageCreateInfo resImageInfo = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    resImageInfo.imageType         = VK_IMAGE_TYPE_2D;
    resImageInfo.format            = VK_FORMAT_R8G8B8A8_UNORM;
    resImageInfo.extent.width      = winWidth;
    resImageInfo.extent.height     = winHeight;
    resImageInfo.extent.depth      = 1;
    resImageInfo.mipLevels         = 1;
    resImageInfo.arrayLayers       = 1;
    resImageInfo.samples           = VK_SAMPLE_COUNT_1_BIT;
    resImageInfo.tiling            = VK_IMAGE_TILING_OPTIMAL;
    resImageInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    resImageInfo.flags         = 0;
    resImageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    {
      nvvk::AllocationID allocationId;
      m_framebuffer.imgColorResolved =
          m_framebuffer.memAllocator.createImage(resImageInfo, allocationId, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    }
  }

  // views after allocation handling

  VkImageViewCreateInfo cbImageViewInfo           = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  cbImageViewInfo.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
  cbImageViewInfo.format                          = cbImageInfo.format;
  cbImageViewInfo.components.r                    = VK_COMPONENT_SWIZZLE_R;
  cbImageViewInfo.components.g                    = VK_COMPONENT_SWIZZLE_G;
  cbImageViewInfo.components.b                    = VK_COMPONENT_SWIZZLE_B;
  cbImageViewInfo.components.a                    = VK_COMPONENT_SWIZZLE_A;
  cbImageViewInfo.flags                           = 0;
  cbImageViewInfo.subresourceRange.levelCount     = 1;
  cbImageViewInfo.subresourceRange.baseMipLevel   = 0;
  cbImageViewInfo.subresourceRange.layerCount     = 1;
  cbImageViewInfo.subresourceRange.baseArrayLayer = 0;
  cbImageViewInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;

  cbImageViewInfo.image = m_framebuffer.imgColor;
  result                = vkCreateImageView(m_device, &cbImageViewInfo, NULL, &m_framebuffer.viewColor);
  assert(result == VK_SUCCESS);

  if(m_framebuffer.useResolved)
  {
    cbImageViewInfo.image = m_framebuffer.imgColorResolved;
    result                = vkCreateImageView(m_device, &cbImageViewInfo, NULL, &m_framebuffer.viewColorResolved);
    assert(result == VK_SUCCESS);
  }

  VkImageViewCreateInfo dsImageViewInfo           = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  dsImageViewInfo.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
  dsImageViewInfo.format                          = dsImageInfo.format;
  dsImageViewInfo.components.r                    = VK_COMPONENT_SWIZZLE_R;
  dsImageViewInfo.components.g                    = VK_COMPONENT_SWIZZLE_G;
  dsImageViewInfo.components.b                    = VK_COMPONENT_SWIZZLE_B;
  dsImageViewInfo.components.a                    = VK_COMPONENT_SWIZZLE_A;
  dsImageViewInfo.flags                           = 0;
  dsImageViewInfo.subresourceRange.levelCount     = 1;
  dsImageViewInfo.subresourceRange.baseMipLevel   = 0;
  dsImageViewInfo.subresourceRange.layerCount     = 1;
  dsImageViewInfo.subresourceRange.baseArrayLayer = 0;
  dsImageViewInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_STENCIL_BIT | VK_IMAGE_ASPECT_DEPTH_BIT;

  dsImageViewInfo.image = m_framebuffer.imgDepthStencil;
  result                = vkCreateImageView(m_device, &dsImageViewInfo, NULL, &m_framebuffer.viewDepthStencil);
  assert(result == VK_SUCCESS);
  // initial resource transitions
  {
    VkCommandBuffer cmd = createTempCmdBuffer();

#if !HAS_OPENGL
    m_swapChain->cmdUpdateBarriers(cmd);
#endif

    cmdImageTransition(cmd, m_framebuffer.imgColor, VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_ACCESS_TRANSFER_READ_BIT,
                       VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

    cmdImageTransition(cmd, m_framebuffer.imgDepthStencil, VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, 0,
                       VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                       VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

    if(m_framebuffer.useResolved)
    {
      cmdImageTransition(cmd, m_framebuffer.imgColorResolved, VK_IMAGE_ASPECT_COLOR_BIT, 0,
                         VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    }

    vkEndCommandBuffer(cmd);

    submissionEnqueue(cmd);
    submissionExecute();
    synchronize();
    resetTempResources();
  }

  {
    // Create framebuffers
    VkImageView bindInfos[2];
    bindInfos[0] = m_framebuffer.viewColor;
    bindInfos[1] = m_framebuffer.viewDepthStencil;

    VkFramebuffer           fb;
    VkFramebufferCreateInfo fbInfo = {VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    fbInfo.attachmentCount         = NV_ARRAY_SIZE(bindInfos);
    fbInfo.pAttachments            = bindInfos;
    fbInfo.width                   = m_framebuffer.renderWidth;
    fbInfo.height                  = m_framebuffer.renderHeight;
    fbInfo.layers                  = 1;

    fbInfo.renderPass = m_framebuffer.passClear;
    result            = vkCreateFramebuffer(m_device, &fbInfo, NULL, &fb);
    assert(result == VK_SUCCESS);
    m_framebuffer.fboScene = fb;
  }


  // ui related
  {
    VkImageView uiTarget = m_framebuffer.useResolved ? m_framebuffer.viewColorResolved : m_framebuffer.viewColor;

    // Create framebuffers
    VkImageView bindInfos[1];
    bindInfos[0] = uiTarget;

    VkFramebuffer           fb;
    VkFramebufferCreateInfo fbInfo = {VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    fbInfo.attachmentCount         = NV_ARRAY_SIZE(bindInfos);
    fbInfo.pAttachments            = bindInfos;
    fbInfo.width                   = winWidth;
    fbInfo.height                  = winHeight;
    fbInfo.layers                  = 1;

    fbInfo.renderPass = m_framebuffer.passUI;
    result            = vkCreateFramebuffer(m_device, &fbInfo, NULL, &fb);
    assert(result == VK_SUCCESS);
    m_framebuffer.fboUI = fb;
  }

  {
    VkViewport vp;
    VkRect2D   sc;
    vp.x        = 0;
    vp.y        = 0;
    vp.width    = float(m_framebuffer.renderWidth);
    vp.height   = float(m_framebuffer.renderHeight);
    vp.minDepth = 0.0f;
    vp.maxDepth = 1.0f;

    sc.offset.x      = 0;
    sc.offset.y      = 0;
    sc.extent.width  = m_framebuffer.renderWidth;
    sc.extent.height = m_framebuffer.renderHeight;

    m_framebuffer.viewport = vp;
    m_framebuffer.scissor  = sc;

    vp.width         = float(winWidth);
    vp.height        = float(winHeight);
    sc.extent.width  = winWidth;
    sc.extent.height = winHeight;

    m_framebuffer.viewportUI = vp;
    m_framebuffer.scissorUI  = sc;
  }


  if(m_framebuffer.msaa != oldMsaa)
  {
    ImGui::ReInitPipelinesVK(m_framebuffer.passUI);
  }
  if(m_framebuffer.msaa != oldMsaa && hasPipes())
  {
    // reinit pipelines
    initPipes();
  }

  return true;
}

void ResourcesVK::deinitFramebuffer()
{
  synchronize();

  vkDestroyImageView(m_device, m_framebuffer.viewColor, nullptr);
  vkDestroyImageView(m_device, m_framebuffer.viewDepthStencil, nullptr);
  m_framebuffer.viewColor        = VK_NULL_HANDLE;
  m_framebuffer.viewDepthStencil = VK_NULL_HANDLE;

  vkDestroyImage(m_device, m_framebuffer.imgColor, nullptr);
  vkDestroyImage(m_device, m_framebuffer.imgDepthStencil, nullptr);
  m_framebuffer.imgColor        = VK_NULL_HANDLE;
  m_framebuffer.imgDepthStencil = VK_NULL_HANDLE;

  if(m_framebuffer.imgColorResolved)
  {
    vkDestroyImageView(m_device, m_framebuffer.viewColorResolved, nullptr);
    m_framebuffer.viewColorResolved = VK_NULL_HANDLE;

    vkDestroyImage(m_device, m_framebuffer.imgColorResolved, nullptr);
    m_framebuffer.imgColorResolved = VK_NULL_HANDLE;
  }

  vkDestroyFramebuffer(m_device, m_framebuffer.fboScene, nullptr);
  m_framebuffer.fboScene = VK_NULL_HANDLE;

  vkDestroyFramebuffer(m_device, m_framebuffer.fboUI, nullptr);
  m_framebuffer.fboUI = VK_NULL_HANDLE;

  m_framebuffer.memAllocator.freeAll();
  m_framebuffer.memAllocator.deinit();
}

void ResourcesVK::initPipes()
{
  VkResult result;

  m_pipeChangeID++;

  if(hasPipes())
  {
    deinitPipes();
  }

  VkSampleCountFlagBits samplesUsed = getSampleCountFlagBits(m_framebuffer.msaa);

  // Create static state info for the pipeline.
  VkVertexInputBindingDescription vertexBinding[2];
  vertexBinding[0].stride    = m_vertexSize;
  vertexBinding[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
  vertexBinding[0].binding   = 0;
  vertexBinding[1].stride    = m_vertexAttributeSize;
  vertexBinding[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
  vertexBinding[1].binding   = 1;

  std::vector<VkVertexInputAttributeDescription> attributes;
  attributes.resize(2 + m_extraAttributes);
  attributes[0].location = VERTEX_POS;
  attributes[0].binding  = 0;
  attributes[0].format   = m_fp16 ? VK_FORMAT_R16G16B16_SFLOAT : VK_FORMAT_R32G32B32_SFLOAT;
  attributes[0].offset   = m_fp16 ? offsetof(CadScene::VertexFP16, position) : offsetof(CadScene::Vertex, position);
  attributes[1].location = VERTEX_NORMAL;
  attributes[1].binding  = 1;
  attributes[1].format   = m_fp16 ? VK_FORMAT_R16G16B16_SFLOAT : VK_FORMAT_R32G32B32_SFLOAT;
  attributes[1].offset = m_fp16 ? offsetof(CadScene::VertexAttributesFP16, normal) : offsetof(CadScene::VertexAttributes, normal);
  for(uint32_t i = 0; i < m_extraAttributes; i++)
  {
    attributes[2 + i].location = VERTEX_XTRA + i;
    attributes[2 + i].binding  = 1;
    attributes[2 + i].format   = m_fp16 ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_R32G32B32A32_SFLOAT;
    attributes[2 + i].offset   = m_fp16 ? (sizeof(CadScene::VertexAttributesFP16) + sizeof(half) * 4 * i) :
                                        (sizeof(CadScene::VertexAttributes) + sizeof(float) * 4 * i);
  }

  VkPipelineVertexInputStateCreateInfo viStateInfo = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
  viStateInfo.vertexBindingDescriptionCount        = NV_ARRAY_SIZE(vertexBinding);
  viStateInfo.pVertexBindingDescriptions           = vertexBinding;
  viStateInfo.vertexAttributeDescriptionCount      = uint32_t(attributes.size());
  viStateInfo.pVertexAttributeDescriptions         = attributes.data();

  VkPipelineInputAssemblyStateCreateInfo iaStateInfo = {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
  iaStateInfo.topology                               = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  iaStateInfo.primitiveRestartEnable                 = VK_FALSE;

  VkPipelineViewportStateCreateInfo vpStateInfo = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
  vpStateInfo.viewportCount                     = 1;
  vpStateInfo.scissorCount                      = 1;

  VkPipelineRasterizationStateCreateInfo rsStateInfo = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
  rsStateInfo.rasterizerDiscardEnable                = VK_FALSE;
  rsStateInfo.polygonMode                            = VK_POLYGON_MODE_FILL;
  rsStateInfo.cullMode                               = m_cullBackFace ? VK_CULL_MODE_BACK_BIT : VK_CULL_MODE_NONE;
  rsStateInfo.frontFace                              = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  rsStateInfo.depthClampEnable                       = VK_TRUE;
  rsStateInfo.depthBiasEnable                        = VK_FALSE;
  rsStateInfo.depthBiasConstantFactor                = 0.0;
  rsStateInfo.depthBiasSlopeFactor                   = 0.0f;
  rsStateInfo.depthBiasClamp                         = 0.0f;
  rsStateInfo.lineWidth                              = float(m_framebuffer.supersample);

  VkPipelineRasterizationStateCreateInfo rsStateInfoBbox = rsStateInfo;
  rsStateInfoBbox.polygonMode                            = VK_POLYGON_MODE_LINE;
  rsStateInfoBbox.cullMode                               = VK_CULL_MODE_NONE;

  // create a color blend attachment that does blending
  VkPipelineColorBlendAttachmentState cbAttachmentState[1] = {};
  cbAttachmentState[0].blendEnable                         = VK_FALSE;
  cbAttachmentState[0].colorWriteMask =
      VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

  // create a color blend state that does blending
  VkPipelineColorBlendStateCreateInfo cbStateInfo = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
  cbStateInfo.logicOpEnable                       = VK_FALSE;
  cbStateInfo.attachmentCount                     = 1;
  cbStateInfo.pAttachments                        = cbAttachmentState;
  cbStateInfo.blendConstants[0]                   = 1.0f;
  cbStateInfo.blendConstants[1]                   = 1.0f;
  cbStateInfo.blendConstants[2]                   = 1.0f;
  cbStateInfo.blendConstants[3]                   = 1.0f;

  VkPipelineDepthStencilStateCreateInfo dsStateInfo = {VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
  dsStateInfo.depthTestEnable                       = VK_TRUE;
  dsStateInfo.depthWriteEnable                      = VK_TRUE;
  dsStateInfo.depthCompareOp                        = VK_COMPARE_OP_LESS;
  dsStateInfo.depthBoundsTestEnable                 = VK_FALSE;
  dsStateInfo.stencilTestEnable                     = VK_FALSE;
  dsStateInfo.minDepthBounds                        = 0.0f;
  dsStateInfo.maxDepthBounds                        = 1.0f;

  VkPipelineMultisampleStateCreateInfo msStateInfo = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
  msStateInfo.rasterizationSamples                 = samplesUsed;
  msStateInfo.sampleShadingEnable                  = VK_FALSE;
  msStateInfo.minSampleShading                     = 1.0f;
  uint32_t sampleMask                              = 0xFFFFFFFF;
  msStateInfo.pSampleMask                          = &sampleMask;

  VkPipelineTessellationStateCreateInfo tessStateInfo = {VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO};
  tessStateInfo.patchControlPoints                    = 0;

  VkPipelineDynamicStateCreateInfo dynStateInfo = {VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
  VkDynamicState                   dynStates[]  = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
  dynStateInfo.dynamicStateCount                = NV_ARRAY_SIZE(dynStates);
  dynStateInfo.pDynamicStates                   = dynStates;

  for(int mode = 0; mode < (m_nativeMeshSupport ? NUM_MODES : MODE_BBOX + 1); mode++)
  {
    VkPipeline                   pipeline;
    VkGraphicsPipelineCreateInfo pipelineInfo = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pipelineInfo.pVertexInputState            = &viStateInfo;
    pipelineInfo.pInputAssemblyState          = &iaStateInfo;
    pipelineInfo.pViewportState               = &vpStateInfo;
    pipelineInfo.pRasterizationState          = (mode == MODE_BBOX) ? &rsStateInfoBbox : &rsStateInfo;
    pipelineInfo.pColorBlendState             = &cbStateInfo;
    pipelineInfo.pDepthStencilState           = &dsStateInfo;
    pipelineInfo.pMultisampleState            = &msStateInfo;
    pipelineInfo.pTessellationState           = &tessStateInfo;
    pipelineInfo.pDynamicState                = &dynStateInfo;

    pipelineInfo.renderPass = m_framebuffer.passPreserve;
    pipelineInfo.subpass    = 0;

    switch(mode)
    {
      case MODE_REGULAR:
        pipelineInfo.layout  = m_setupRegular.container.getPipeLayout();
        iaStateInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        break;
      case MODE_BBOX:
        pipelineInfo.layout                         = m_setupBbox.container.getPipeLayout();
        iaStateInfo.topology                        = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
        viStateInfo.vertexBindingDescriptionCount   = 0;
        viStateInfo.vertexAttributeDescriptionCount = 0;
        viStateInfo.pVertexAttributeDescriptions    = nullptr;
        viStateInfo.pVertexBindingDescriptions      = nullptr;
        break;
      case MODE_MESH:
        iaStateInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        pipelineInfo.layout  = m_setupMeshTask.container.getPipeLayout();
        break;
      case MODE_TASK_MESH:
        iaStateInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        pipelineInfo.layout  = m_setupMeshTask.container.getPipeLayout();
        break;
    }

    VkPipelineShaderStageCreateInfo stages[3];
    memset(stages, 0, sizeof(stages));
    pipelineInfo.pStages = stages;

    VkPipelineShaderStageCreateInfo& stage0 = stages[0];
    stage0.sType                            = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage0.pName                            = "main";

    VkPipelineShaderStageCreateInfo& stage1 = stages[1];
    stage1.sType                            = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage1.pName                            = "main";

    VkPipelineShaderStageCreateInfo& stage2 = stages[2];
    stage2.sType                            = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage2.pName                            = "main";

    switch(mode)
    {
      case MODE_REGULAR:
        pipelineInfo.stageCount = 2;
        stage0.stage            = VK_SHADER_STAGE_VERTEX_BIT;
        stage0.module           = m_shaderManager.get(m_shaders.object_vertex);
        stage1.stage            = VK_SHADER_STAGE_FRAGMENT_BIT;
        stage1.module           = m_shaderManager.get(m_shaders.object_fragment);
        break;
      case MODE_BBOX:
        pipelineInfo.stageCount = 3;
        stage0.stage            = VK_SHADER_STAGE_VERTEX_BIT;
        stage0.module           = m_shaderManager.get(m_shaders.bbox_vertex);
        stage1.stage            = VK_SHADER_STAGE_GEOMETRY_BIT;
        stage1.module           = m_shaderManager.get(m_shaders.bbox_geometry);
        stage2.stage            = VK_SHADER_STAGE_FRAGMENT_BIT;
        stage2.module           = m_shaderManager.get(m_shaders.bbox_fragment);
        break;
      case MODE_MESH:
        pipelineInfo.stageCount = 2;
        stage0.stage            = VK_SHADER_STAGE_MESH_BIT_NV;
        stage0.module           = m_shaderManager.get(m_shaders.object_mesh);
        stage1.stage            = VK_SHADER_STAGE_FRAGMENT_BIT;
        stage1.module           = m_shaderManager.get(m_shaders.object_fragment);
        break;
      case MODE_TASK_MESH:
        pipelineInfo.stageCount = 3;
        stage0.stage            = VK_SHADER_STAGE_TASK_BIT_NV;
        stage0.module           = m_shaderManager.get(m_shaders.object_task);
        stage1.stage            = VK_SHADER_STAGE_MESH_BIT_NV;
        stage1.module           = m_shaderManager.get(m_shaders.object_task_mesh);
        stage2.stage            = VK_SHADER_STAGE_FRAGMENT_BIT;
        stage2.module           = m_shaderManager.get(m_shaders.object_fragment);
        break;
    }

    result = vkCreateGraphicsPipelines(m_device, VK_NULL_HANDLE, 1, &pipelineInfo, NULL, &pipeline);
    assert(result == VK_SUCCESS);

    switch(mode)
    {
      case MODE_REGULAR:
        m_setupRegular.pipeline = pipeline;
        break;
      case MODE_BBOX:
        m_setupBbox.pipeline = pipeline;
        break;
      case MODE_MESH:
        m_setupMeshTask.pipelineNoTask = pipeline;
        break;
      case MODE_TASK_MESH:
        m_setupMeshTask.pipeline = pipeline;
        break;
    }
  }
}

void ResourcesVK::deinitPipes()
{
  vkDestroyPipeline(m_device, m_setupRegular.pipeline, NULL);
  m_setupRegular.pipeline = NULL;
  vkDestroyPipeline(m_device, m_setupBbox.pipeline, NULL);
  m_setupBbox.pipeline = NULL;
  if(m_nativeMeshSupport)
  {
    vkDestroyPipeline(m_device, m_setupMeshTask.pipeline, NULL);
    m_setupMeshTask.pipeline = NULL;
    vkDestroyPipeline(m_device, m_setupMeshTask.pipelineNoTask, NULL);
    m_setupMeshTask.pipelineNoTask = NULL;
  }
}

void ResourcesVK::cmdDynamicState(VkCommandBuffer cmd) const
{
  vkCmdSetViewport(cmd, 0, 1, &m_framebuffer.viewport);
  vkCmdSetScissor(cmd, 0, 1, &m_framebuffer.scissor);
}

void ResourcesVK::cmdBeginRenderPass(VkCommandBuffer cmd, bool clear, bool hasSecondary) const
{
  VkRenderPassBeginInfo renderPassBeginInfo    = {VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
  renderPassBeginInfo.renderPass               = clear ? m_framebuffer.passClear : m_framebuffer.passPreserve;
  renderPassBeginInfo.framebuffer              = m_framebuffer.fboScene;
  renderPassBeginInfo.renderArea.offset.x      = 0;
  renderPassBeginInfo.renderArea.offset.y      = 0;
  renderPassBeginInfo.renderArea.extent.width  = m_framebuffer.renderWidth;
  renderPassBeginInfo.renderArea.extent.height = m_framebuffer.renderHeight;
  renderPassBeginInfo.clearValueCount          = 2;
  VkClearValue clearValues[2];
  clearValues[0].color.float32[0]     = 0.2f;
  clearValues[0].color.float32[1]     = 0.2f;
  clearValues[0].color.float32[2]     = 0.2f;
  clearValues[0].color.float32[3]     = 0.0f;
  clearValues[1].depthStencil.depth   = 1.0f;
  clearValues[1].depthStencil.stencil = 0;
  renderPassBeginInfo.pClearValues    = clearValues;
  vkCmdBeginRenderPass(cmd, &renderPassBeginInfo,
                       hasSecondary ? VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS : VK_SUBPASS_CONTENTS_INLINE);
}

void ResourcesVK::cmdPipelineBarrier(VkCommandBuffer cmd) const
{
  // color transition
  {
    VkImageSubresourceRange colorRange;
    memset(&colorRange, 0, sizeof(colorRange));
    colorRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    colorRange.baseMipLevel   = 0;
    colorRange.levelCount     = VK_REMAINING_MIP_LEVELS;
    colorRange.baseArrayLayer = 0;
    colorRange.layerCount     = 1;

    VkImageMemoryBarrier memBarrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    memBarrier.srcAccessMask        = VK_ACCESS_TRANSFER_READ_BIT;
    memBarrier.dstAccessMask        = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    memBarrier.oldLayout            = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    memBarrier.newLayout            = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    memBarrier.image                = m_framebuffer.imgColor;
    memBarrier.subresourceRange     = colorRange;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_FALSE,
                         0, NULL, 0, NULL, 1, &memBarrier);
  }

  // Prepare the depth+stencil for reading.

  {
    VkImageSubresourceRange depthStencilRange;
    memset(&depthStencilRange, 0, sizeof(depthStencilRange));
    depthStencilRange.aspectMask     = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
    depthStencilRange.baseMipLevel   = 0;
    depthStencilRange.levelCount     = VK_REMAINING_MIP_LEVELS;
    depthStencilRange.baseArrayLayer = 0;
    depthStencilRange.layerCount     = 1;

    VkImageMemoryBarrier memBarrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    memBarrier.sType                = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    memBarrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    memBarrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    memBarrier.oldLayout     = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    memBarrier.newLayout     = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    memBarrier.image         = m_framebuffer.imgDepthStencil;
    memBarrier.subresourceRange = depthStencilRange;

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
                         VK_FALSE, 0, NULL, 0, NULL, 1, &memBarrier);
  }
}


void ResourcesVK::cmdImageTransition(VkCommandBuffer    cmd,
                                     VkImage            img,
                                     VkImageAspectFlags aspects,
                                     VkAccessFlags      src,
                                     VkAccessFlags      dst,
                                     VkImageLayout      oldLayout,
                                     VkImageLayout      newLayout) const
{

  VkPipelineStageFlags srcPipe = nvvk::makeAccessMaskPipelineStageFlags(src);
  VkPipelineStageFlags dstPipe = nvvk::makeAccessMaskPipelineStageFlags(dst);

  VkImageSubresourceRange range;
  memset(&range, 0, sizeof(range));
  range.aspectMask     = aspects;
  range.baseMipLevel   = 0;
  range.levelCount     = VK_REMAINING_MIP_LEVELS;
  range.baseArrayLayer = 0;
  range.layerCount     = VK_REMAINING_ARRAY_LAYERS;

  VkImageMemoryBarrier memBarrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  memBarrier.sType                = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  memBarrier.dstAccessMask        = dst;
  memBarrier.srcAccessMask        = src;
  memBarrier.oldLayout            = oldLayout;
  memBarrier.newLayout            = newLayout;
  memBarrier.image                = img;
  memBarrier.subresourceRange     = range;

  vkCmdPipelineBarrier(cmd, srcPipe, dstPipe, VK_FALSE, 0, NULL, 0, NULL, 1, &memBarrier);
}

VkCommandBuffer ResourcesVK::createCmdBuffer(VkCommandPool pool, bool singleshot, bool primary, bool secondaryInClear) const
{
  VkResult result;
  bool     secondary = !primary;

  // Create the command buffer.
  VkCommandBufferAllocateInfo cmdInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cmdInfo.commandPool                 = pool;
  cmdInfo.level                       = primary ? VK_COMMAND_BUFFER_LEVEL_PRIMARY : VK_COMMAND_BUFFER_LEVEL_SECONDARY;
  cmdInfo.commandBufferCount          = 1;
  VkCommandBuffer cmd;
  result = vkAllocateCommandBuffers(m_device, &cmdInfo, &cmd);
  assert(result == VK_SUCCESS);

  cmdBegin(cmd, singleshot, primary, secondaryInClear);

  return cmd;
}

VkCommandBuffer ResourcesVK::createTempCmdBuffer(bool primary /*=true*/, bool secondaryInClear /*=false*/)
{
  VkCommandBuffer cmd =
      m_ringCmdPool.createCommandBuffer(primary ? VK_COMMAND_BUFFER_LEVEL_PRIMARY : VK_COMMAND_BUFFER_LEVEL_SECONDARY, false);
  cmdBegin(cmd, true, primary, secondaryInClear);
  return cmd;
}


void ResourcesVK::cmdBegin(VkCommandBuffer cmd, bool singleshot, bool primary, bool secondaryInClear) const
{
  VkResult result;
  bool     secondary = !primary;

  VkCommandBufferInheritanceInfo inheritInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO};
  if(secondary)
  {
    inheritInfo.renderPass  = secondaryInClear ? m_framebuffer.passClear : m_framebuffer.passPreserve;
    inheritInfo.framebuffer = m_framebuffer.fboScene;
  }

  VkCommandBufferBeginInfo beginInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  // the sample is resubmitting re-use commandbuffers to the queue while they may still be executed by GPU
  // we only use fences to prevent deleting commandbuffers that are still in flight
  beginInfo.flags = singleshot ? VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT : VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
  // the sample's secondary buffers always are called within passes as they contain drawcalls
  beginInfo.flags |= secondary ? VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT : 0;
  beginInfo.pInheritanceInfo = &inheritInfo;

  result = vkBeginCommandBuffer(cmd, &beginInfo);
  assert(result == VK_SUCCESS);
}

void ResourcesVK::resetTempResources()
{
  synchronize();
  m_ringFences.reset();
  m_ringCmdPool.reset();
}

bool ResourcesVK::initScene(const CadScene& cadscene)
{
  m_fp16                = cadscene.m_cfg.fp16;
  m_extraAttributes     = cadscene.m_cfg.extraAttributes;
  m_vertexSize          = (uint32_t)cadscene.getVertexSize();
  m_vertexAttributeSize = (uint32_t)cadscene.getVertexAttributeSize();

  m_scene.init(cadscene, m_device, m_physical, m_queue, m_queueFamily);


  uint32_t geometryBindings =
      USE_PER_GEOMETRY_VIEWS ? uint32_t(m_scene.m_geometry.size()) : uint32_t(m_scene.m_geometryMem.getChunkCount() * 2);

  {
    // Allocation phase

    {
      m_setupRegular.container.at(DSET_SCENE).initPool(1);
      m_setupRegular.container.at(DSET_OBJECT).initPool(1);


      m_setupBbox.container.at(DSET_SCENE).initPool(1);
      m_setupBbox.container.at(DSET_OBJECT).initPool(1);
      m_setupBbox.container.at(DSET_GEOMETRY).initPool(geometryBindings);
    }
    if(m_nativeMeshSupport)
    {
      m_setupMeshTask.container.at(DSET_SCENE).initPool(1);
      m_setupMeshTask.container.at(DSET_OBJECT).initPool(1);
      m_setupMeshTask.container.at(DSET_GEOMETRY).initPool(geometryBindings);
    }
  }

  {
    // Fill phase
    {
      {
        VkWriteDescriptorSet updateDescriptors[] = {
            m_setupRegular.container.at(DSET_SCENE).makeWrite(0, SCENE_UBO_VIEW, &m_common.viewInfo),
            m_setupRegular.container.at(DSET_OBJECT).makeWrite(0, 0, &m_scene.m_infos.matricesSingle),
            m_setupBbox.container.at(DSET_SCENE).makeWrite(0, SCENE_UBO_VIEW, &m_common.viewInfo),
            m_setupBbox.container.at(DSET_OBJECT).makeWrite(0, 0, &m_scene.m_infos.matricesSingle),

        };
        vkUpdateDescriptorSets(m_device, NV_ARRAY_SIZE(updateDescriptors), updateDescriptors, 0, 0);
      }

      if(m_nativeMeshSupport)
      {
        VkWriteDescriptorSet updateDescriptors[] = {
            m_setupMeshTask.container.at(DSET_SCENE).makeWrite(0, SCENE_UBO_VIEW, &m_common.viewInfo),
            m_setupMeshTask.container.at(DSET_SCENE).makeWrite(0, SCENE_SSBO_STATS, &m_common.statsInfo),
            m_setupMeshTask.container.at(DSET_OBJECT).makeWrite(0, 0, &m_scene.m_infos.matricesSingle),
        };
        vkUpdateDescriptorSets(m_device, NV_ARRAY_SIZE(updateDescriptors), updateDescriptors, 0, 0);
      }
    }
#if USE_PER_GEOMETRY_VIEWS
    {
      std::vector<VkWriteDescriptorSet> writeUpdates;

      for(uint32_t g = 0; g < m_scene.m_geometry.size(); g++)
      {
        CadSceneVK::Geometry& geom = m_scene.m_geometry[g];

        if(!geom.meshletDesc.range)
          continue;

        writeUpdates.push_back(m_setupBbox.container.at(DSET_GEOMETRY).makeWrite(g, GEOMETRY_SSBO_MESHLETDESC, &geom.meshletDesc));
        writeUpdates.push_back(m_setupBbox.container.at(DSET_GEOMETRY).makeWrite(g, GEOMETRY_SSBO_PRIM, &geom.meshletPrim));
        writeUpdates.push_back(m_setupBbox.container.at(DSET_GEOMETRY).makeWrite(g, GEOMETRY_TEX_VBO, &geom.vboView));
        writeUpdates.push_back(m_setupBbox.container.at(DSET_GEOMETRY).makeWrite(g, GEOMETRY_TEX_ABO, &geom.aboView));
        writeUpdates.push_back(m_setupBbox.container.at(DSET_GEOMETRY).makeWrite(g, GEOMETRY_TEX_IBO, &geom.vertView));


        if(m_nativeMeshSupport)
        {
          writeUpdates.push_back(m_setupMeshTask.container.at(DSET_GEOMETRY).makeWrite(g, GEOMETRY_SSBO_MESHLETDESC, &geom.meshletDesc));
          writeUpdates.push_back(m_setupMeshTask.container.at(DSET_GEOMETRY).makeWrite(g, GEOMETRY_SSBO_PRIM, &geom.meshletPrim));
          writeUpdates.push_back(m_setupMeshTask.container.at(DSET_GEOMETRY).makeWrite(g, GEOMETRY_TEX_VBO, &geom.vboView));
          writeUpdates.push_back(m_setupMeshTask.container.at(DSET_GEOMETRY).makeWrite(g, GEOMETRY_TEX_ABO, &geom.aboView));
          writeUpdates.push_back(m_setupMeshTask.container.at(DSET_GEOMETRY).makeWrite(g, GEOMETRY_TEX_IBO, &geom.vertView));
        }
      }

      vkUpdateDescriptorSets(m_device, (uint32_t)writeUpdates.size(), writeUpdates.data(), 0, 0);
    }
#else
    {
      std::vector<VkWriteDescriptorSet> writeUpdates;

      for(VkDeviceSize g = 0; g < m_scene.m_geometryMem.getChunkCount(); g++)
      {
        const auto& chunk = m_scene.m_geometryMem.getChunk(g);


        writeUpdates.push_back(
            m_setupBbox.container.at(DSET_GEOMETRY).makeWrite(g * 2 + 0, GEOMETRY_SSBO_MESHLETDESC, &chunk.meshInfo));

        writeUpdates.push_back(m_setupBbox.container.at(DSET_GEOMETRY).makeWrite(g * 2 + 0, GEOMETRY_SSBO_PRIM, &chunk.meshIndicesInfo));
        writeUpdates.push_back(m_setupBbox.container.at(DSET_GEOMETRY).makeWrite(g * 2 + 0, GEOMETRY_TEX_VBO, &chunk.vboView));
        writeUpdates.push_back(m_setupBbox.container.at(DSET_GEOMETRY).makeWrite(g * 2 + 0, GEOMETRY_TEX_ABO, &chunk.aboView));
        writeUpdates.push_back(m_setupBbox.container.at(DSET_GEOMETRY).makeWrite(g * 2 + 0, GEOMETRY_TEX_IBO, &chunk.vert32View));

        writeUpdates.push_back(
            m_setupBbox.container.at(DSET_GEOMETRY).makeWrite(g * 2 + 1, GEOMETRY_SSBO_MESHLETDESC, &chunk.meshInfo));

        writeUpdates.push_back(m_setupBbox.container.at(DSET_GEOMETRY).makeWrite(g * 2 + 1, GEOMETRY_SSBO_PRIM, &chunk.meshIndicesInfo));
        writeUpdates.push_back(m_setupBbox.container.at(DSET_GEOMETRY).makeWrite(g * 2 + 1, GEOMETRY_TEX_VBO, &chunk.vboView));
        writeUpdates.push_back(m_setupBbox.container.at(DSET_GEOMETRY).makeWrite(g * 2 + 1, GEOMETRY_TEX_ABO, &chunk.aboView));
        writeUpdates.push_back(m_setupBbox.container.at(DSET_GEOMETRY).makeWrite(g * 2 + 1, GEOMETRY_TEX_IBO, &chunk.vert16View));


        if(m_nativeMeshSupport)
        {
          writeUpdates.push_back(
              m_setupMeshTask.container.at(DSET_GEOMETRY).makeWrite(g * 2 + 0, GEOMETRY_SSBO_MESHLETDESC, &chunk.meshInfo));
          writeUpdates.push_back(
              m_setupMeshTask.container.at(DSET_GEOMETRY).makeWrite(g * 2 + 0, GEOMETRY_SSBO_PRIM, &chunk.meshIndicesInfo));
          writeUpdates.push_back(
              m_setupMeshTask.container.at(DSET_GEOMETRY).makeWrite(g * 2 + 0, GEOMETRY_TEX_VBO, &chunk.vboView));
          writeUpdates.push_back(
              m_setupMeshTask.container.at(DSET_GEOMETRY).makeWrite(g * 2 + 0, GEOMETRY_TEX_ABO, &chunk.aboView));
          writeUpdates.push_back(
              m_setupMeshTask.container.at(DSET_GEOMETRY).makeWrite(g * 2 + 0, GEOMETRY_TEX_IBO, &chunk.vert32View));


          writeUpdates.push_back(
              m_setupMeshTask.container.at(DSET_GEOMETRY).makeWrite(g * 2 + 1, GEOMETRY_SSBO_MESHLETDESC, &chunk.meshInfo));
          writeUpdates.push_back(
              m_setupMeshTask.container.at(DSET_GEOMETRY).makeWrite(g * 2 + 1, GEOMETRY_SSBO_PRIM, &chunk.meshIndicesInfo));
          writeUpdates.push_back(
              m_setupMeshTask.container.at(DSET_GEOMETRY).makeWrite(g * 2 + 1, GEOMETRY_TEX_VBO, &chunk.vboView));
          writeUpdates.push_back(
              m_setupMeshTask.container.at(DSET_GEOMETRY).makeWrite(g * 2 + 1, GEOMETRY_TEX_ABO, &chunk.aboView));
          writeUpdates.push_back(
              m_setupMeshTask.container.at(DSET_GEOMETRY).makeWrite(g * 2 + 1, GEOMETRY_TEX_IBO, &chunk.vert16View));
        }
      }

      vkUpdateDescriptorSets(m_device, (uint32_t)writeUpdates.size(), writeUpdates.data(), 0, 0);
    }
#endif
  }

  // fp16/
  initPipes();

  return true;
}

void ResourcesVK::deinitScene()
{
  // guard by synchronization as some stuff is unsafe to delete while in use
  synchronize();

  m_scene.deinit();

  m_setupRegular.container.deinitPools();
  m_setupBbox.container.deinitPools();

  if(m_nativeMeshSupport)
  {
    m_setupMeshTask.container.deinitPools();
  }
}

void ResourcesVK::synchronize()
{
  vkDeviceWaitIdle(m_device);
}

nvmath::mat4f ResourcesVK::perspectiveProjection(float fovy, float aspect, float nearPlane, float farPlane) const
{
  // vulkan uses DX style 0,1 z clipspace

  nvmath::mat4f M;
  float         r, l, b, t;
  float         f = farPlane;
  float         n = nearPlane;

  t = n * tanf(fovy * nv_to_rad * (0.5f));
  b = -t;

  l = b * aspect;
  r = t * aspect;

  M.a00 = (2.0f * n) / (r - l);
  M.a10 = 0.0f;
  M.a20 = 0.0f;
  M.a30 = 0.0f;

  M.a01 = 0.0f;
  M.a11 = -(2.0f * n) / (t - b);
  M.a21 = 0.0f;
  M.a31 = 0.0f;

  M.a02 = (r + l) / (r - l);
  M.a12 = (t + b) / (t - b);
  M.a22 = -(f) / (f - n);
  M.a32 = -1.0f;

  M.a03 = 0.0;
  M.a13 = 0.0;
  M.a23 = (f * n) / (n - f);
  M.a33 = 0.0;

  return M;
}

VkCommandBuffer ResourcesVK::createBoundingBoxCmdBuffer(VkCommandPool pool, const class RenderList* NV_RESTRICT list) const
{
  const RenderList::DrawItem* NV_RESTRICT drawItems  = list->m_drawItems.data();
  size_t                                  numItems   = list->m_drawItems.size();
  size_t                                  vertexSize = list->m_scene->getVertexSize();

  const CadScene* NV_RESTRICT scene  = list->m_scene;
  const ResourcesVK* NV_RESTRICT res = this;

  const ResourcesVK::DrawSetup& setup = res->m_setupBbox;

  VkCommandBuffer cmd = res->createCmdBuffer(pool, false, false, true);
  res->cmdDynamicState(cmd);

  int  lastMaterial = -1;
  int  lastGeometry = -1;
  int  lastMatrix   = -1;
  int  lastChunk    = -1;
  bool lastShorts   = false;

  bool first = true;
  for(unsigned int i = 0; i < numItems; i++)
  {
    const RenderList::DrawItem& di = drawItems[i];

    if(first)
    {
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, setup.pipeline);

      if(first)
      {
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, setup.container.getPipeLayout(), DSET_SCENE, 1,
                                setup.container.at(DSET_SCENE).getSets(), 0, nullptr);
      }

      first = false;
    }

    if(lastGeometry != di.geometryIndex)
    {
      const CadSceneVK::Geometry& geovk = m_scene.m_geometry[di.geometryIndex];
      int                         chunk = int(geovk.allocation.chunkIndex);

#if USE_PER_GEOMETRY_VIEWS
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, setup.container.getPipeLayout(), DSET_GEOMETRY, 1,
                              setup.container.at(DSET_GEOMETRY).getSets() + di.geometryIndex, 0, nullptr);
#else
      if(chunk != lastChunk || di.shorts != lastShorts)
      {
        int idx = chunk * 2 + (di.shorts ? 1 : 0);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, setup.container.getPipeLayout(), DSET_GEOMETRY, 1,
                                setup.container.at(DSET_GEOMETRY).getSets() + idx, 0, nullptr);

        lastChunk = chunk;
        lastShorts = di.shorts;
      }

      uint32_t offsets[4] = {uint32_t(geovk.meshletDesc.offset / sizeof(NVMeshlet::MeshletDesc)), 0, 0, 0};
      vkCmdPushConstants(cmd, setup.container.getPipeLayout(), VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(offsets), offsets);
#endif

      lastGeometry = di.geometryIndex;
    }

    if(lastMatrix != di.matrixIndex)
    {
      uint32_t offset = di.matrixIndex * res->m_alignedMatrixSize;
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, setup.container.getPipeLayout(), DSET_OBJECT, 1,
                              setup.container.at(DSET_OBJECT).getSets(), 1, &offset);
      lastMatrix = di.matrixIndex;
    }

    vkCmdDraw(cmd, di.meshlet.count, 1, di.meshlet.offset, 0);
  }

  vkEndCommandBuffer(cmd);

  return cmd;
}


}  // namespace meshlettest
