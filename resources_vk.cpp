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

/* Contact ckubisch@nvidia.com (Christoph Kubisch) for feedback */

#include "backends/imgui_vk_extra.h"

#include "renderer.hpp"
#include "resources_vk.hpp"

#include <algorithm>
#include <nvh/nvprint.hpp>
#include <nvvk/pipeline_vk.hpp>

#include "nvmeshlet_builder.hpp"

namespace meshlettest {


/////////////////////////////////////////////////////////////////////////////////


void ResourcesVK::submissionExecute(VkFence fence, bool useImageReadWait, bool useImageWriteSignals)
{
  if(useImageReadWait && m_submissionWaitForRead)
  {
    VkSemaphore semRead = m_swapChain->getActiveReadSemaphore();
    if(semRead)
    {
      m_submission.enqueueWait(semRead, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
    }
    m_submissionWaitForRead = false;
  }

  if(useImageWriteSignals)
  {
    VkSemaphore semWritten = m_swapChain->getActiveWrittenSemaphore();
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

    ImGui_ImplVulkan_RenderDrawData(global.imguiDrawData, cmd);

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

bool ResourcesVK::init(const nvvk::Context* context, const nvvk::SwapChain* swapChain, nvh::Profiler* profiler)
{
  m_fboChangeID  = 0;
  m_pipeChangeID = 0;

  m_context   = context;
  m_swapChain = swapChain;

  m_device      = m_context->m_device;
  m_physical    = m_context->m_physicalDevice;
  m_queue       = m_context->m_queueGCT.queue;
  m_queueFamily = m_context->m_queueGCT.familyIndex;

  LOGI("Vk device: %s\n", m_context->m_physicalInfo.properties10.deviceName)

  initAlignedSizes((uint32_t)m_context->m_physicalInfo.properties10.limits.minUniformBufferOffsetAlignment);

  m_supportsMeshEXT = m_context->hasDeviceExtension(VK_EXT_MESH_SHADER_EXTENSION_NAME);
  m_supportsMeshNV  = m_context->hasDeviceExtension(VK_NV_MESH_SHADER_EXTENSION_NAME);

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
    initPipeLayouts();
  }

  {
    ImGui::InitVK(m_context->m_device, m_context->m_physicalDevice, m_context->m_queueGCT,
                  m_context->m_queueGCT.familyIndex, m_framebuffer.passUI);
  }

  return true;
}

void ResourcesVK::initPipeLayouts()
{
  ///////////////////////////////////////////////////////////////////////////////////////////
  {
    // REGULAR
    DrawSetup& setup = m_setupStandard;
    setup.container.init(m_device);

    auto& bindingsScene = setup.container.at(DSET_SCENE);
    // UBO SCENE
    bindingsScene.addBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
                             VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr);
    bindingsScene.initLayout();
    // UBO OBJECT
    auto& bindingsObject = setup.container.at(DSET_OBJECT);
    bindingsObject.addBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1,
                              VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr);
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
                             VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_GEOMETRY_BIT, nullptr);
    bindingsScene.initLayout();
    // UBO OBJECT
    auto& bindingsObject = setup.container.at(DSET_OBJECT);
    bindingsObject.addBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1,
                              VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_GEOMETRY_BIT, nullptr);
    bindingsObject.initLayout();
    // UBO GEOMETRY
    auto& bindingsGeometry = setup.container.at(DSET_GEOMETRY);
    bindingsGeometry.addBinding(GEOMETRY_SSBO_MESHLETDESC, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr);
    bindingsGeometry.addBinding(GEOMETRY_SSBO_PRIM, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr);
    bindingsGeometry.addBinding(GEOMETRY_TEX_VBO, VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr);
    bindingsGeometry.addBinding(GEOMETRY_TEX_ABO, VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr);

    bindingsGeometry.initLayout();

    VkPushConstantRange range;
    range.offset     = 0;
    range.size       = sizeof(uint32_t) * 4;
    range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    setup.container.initPipeLayout(0, 3, 1, &range);
  }

  for(uint32_t isNV = 0; isNV < 2; isNV++)
  {
    if((isNV && !m_supportsMeshNV) || (!isNV && !m_supportsMeshEXT))
      continue;

    DrawSetup& setup = isNV ? m_setupMeshNV : m_setupMeshEXT;

    // in theory the binding containers are identical for EXT and NV as
    // NV/EXT use same bits, but given we use a common struct to also
    // store pipelines was easier to just create things twice for
    // this sample.

    VkPipelineStageFlags stageMesh = VK_SHADER_STAGE_MESH_BIT_NV;
    VkPipelineStageFlags stageTask = VK_SHADER_STAGE_TASK_BIT_NV;

    setup.container.init(m_device);
    // UBO SCENE
    auto& bindingsScene = setup.container.at(DSET_SCENE);
    bindingsScene.addBinding(SCENE_UBO_VIEW, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
                             stageTask | stageMesh | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr);
    bindingsScene.addBinding(SCENE_SSBO_STATS, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, stageTask | stageMesh, nullptr);
    bindingsScene.initLayout();
    // UBO OBJECT
    auto& bindingsObject = setup.container.at(DSET_OBJECT);
    bindingsObject.addBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1,
                              stageTask | stageMesh | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr);
    bindingsObject.initLayout();
    // UBO GEOMETRY
    auto& bindingsGeometry = setup.container.at(DSET_GEOMETRY);
    bindingsGeometry.addBinding(GEOMETRY_SSBO_MESHLETDESC, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, stageTask | stageMesh, nullptr);
    bindingsGeometry.addBinding(GEOMETRY_SSBO_PRIM, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, stageMesh, nullptr);
    bindingsGeometry.addBinding(GEOMETRY_TEX_VBO, VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1,
                                stageMesh | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr);
    bindingsGeometry.addBinding(GEOMETRY_TEX_ABO, VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1,
                                stageMesh | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr);
    bindingsGeometry.initLayout();

    VkPushConstantRange ranges[2];
    ranges[0].offset     = 0;
    ranges[0].size       = sizeof(uint32_t) * 8;
    ranges[0].stageFlags = VK_SHADER_STAGE_TASK_BIT_NV | VK_SHADER_STAGE_MESH_BIT_NV;
    setup.container.initPipeLayout(0, 3, 1, ranges);
  }
}

void ResourcesVK::deinit()
{
  synchronize();

  ImGui::ShutdownVK();

  {
    vkDestroyBuffer(m_device, m_common.viewBuffer, nullptr);
    m_memAllocator.free(m_common.viewAID);
    vkDestroyBuffer(m_device, m_common.statsBuffer, nullptr);
    m_memAllocator.free(m_common.statsAID);
    vkDestroyBuffer(m_device, m_common.statsReadBuffer, nullptr);
    m_memAllocator.free(m_common.statsReadAID);
  }


  m_ringFences.deinit();
  m_ringCmdPool.deinit();

  deinitScene();
  deinitFramebuffer();
  deinitPipes();
  deinitPrograms();

  m_profilerVK.deinit();

  vkDestroyRenderPass(m_device, m_framebuffer.passClear, nullptr);
  vkDestroyRenderPass(m_device, m_framebuffer.passPreserve, nullptr);
  vkDestroyRenderPass(m_device, m_framebuffer.passUI, nullptr);

  m_setupStandard.container.deinitLayouts();
  m_setupBbox.container.deinitLayouts();

  for(uint32_t isNV = 0; isNV < 2; isNV++)
  {
    if((isNV && !m_supportsMeshNV) || (!isNV && !m_supportsMeshEXT))
      continue;

    DrawSetup& setup = isNV ? m_setupMeshNV : m_setupMeshEXT;
    setup.container.deinitLayouts();
  }

  m_memAllocator.deinit();
}

bool ResourcesVK::initPrograms(const std::string& path, const std::string& prepend)
{
  // EXT_mesh_shader is only available in Vulkan 1.3, and shaderc complains if we don't pass 1.3
  const bool hasExtMesh = m_context->hasDeviceExtension(VK_EXT_MESH_SHADER_EXTENSION_NAME);

  m_shaderManager.init(m_device, 1, hasExtMesh ? 3 : 2);
  m_shaderManager.m_filetype = nvh::ShaderFileManager::FILETYPE_GLSL;

  m_shaderManager.addDirectory(path);
  m_shaderManager.addDirectory(std::string("GLSL_" PROJECT_NAME));
  m_shaderManager.addDirectory(path + std::string(PROJECT_RELDIRECTORY));

  m_shaderManager.m_prepend = std::string("#define IS_VULKAN 1\n") + prepend;

  ///////////////////////////////////////////////////////////////////////////////////////////
  {
    m_shaders.standard_vertex   = m_shaderManager.createShaderModule(VK_SHADER_STAGE_VERTEX_BIT, "draw.vert.glsl");
    m_shaders.standard_fragment = m_shaderManager.createShaderModule(VK_SHADER_STAGE_FRAGMENT_BIT, "draw.frag.glsl");
  }

  {
    m_shaders.bbox_vertex   = m_shaderManager.createShaderModule(VK_SHADER_STAGE_VERTEX_BIT, "meshletbbox.vert.glsl");
    m_shaders.bbox_geometry = m_shaderManager.createShaderModule(VK_SHADER_STAGE_GEOMETRY_BIT, "meshletbbox.geo.glsl");
    m_shaders.bbox_fragment = m_shaderManager.createShaderModule(VK_SHADER_STAGE_FRAGMENT_BIT, "meshletbbox.frag.glsl");
  }

  for(uint32_t isNV = 0; isNV < 2; isNV++)
  {
    if((isNV && !m_supportsMeshNV) || (!isNV && !m_supportsMeshEXT))
      continue;

    MeshShaderModuleIDs& shaders = isNV ? m_shaders.meshNV : m_shaders.meshEXT;
    std::string          prefix  = isNV ? "drawmeshlet_nv" : "drawmeshlet_ext";

    shaders.mesh      = m_shaderManager.createShaderModule(VK_SHADER_STAGE_MESH_BIT_NV, prefix + "_basic.mesh.glsl",
                                                      "#define USE_TASK_STAGE 0\n");
    shaders.task_mesh = m_shaderManager.createShaderModule(VK_SHADER_STAGE_MESH_BIT_NV, prefix + "_basic.mesh.glsl",
                                                           "#define USE_TASK_STAGE 1\n");
    shaders.cull_mesh = m_shaderManager.createShaderModule(VK_SHADER_STAGE_MESH_BIT_NV, prefix + "_cull.mesh.glsl",
                                                           "#define USE_TASK_STAGE 0\n");
    shaders.cull_task_mesh = m_shaderManager.createShaderModule(VK_SHADER_STAGE_MESH_BIT_NV, prefix + "_cull.mesh.glsl",
                                                                "#define USE_TASK_STAGE 1\n");
    shaders.task = m_shaderManager.createShaderModule(VK_SHADER_STAGE_TASK_BIT_NV, prefix + ".task.glsl", "#define USE_TASK_STAGE 1\n");

    shaders.mesh_fragment = m_shaderManager.createShaderModule(VK_SHADER_STAGE_FRAGMENT_BIT, prefix + ".frag.glsl");
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

VkRenderPass ResourcesVK::createPass(bool clear, int msaa) const
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
  result = vkCreateRenderPass(m_device, &rpInfo, nullptr, &rp);
  assert(result == VK_SUCCESS);
  return rp;
}


VkRenderPass ResourcesVK::createPassUI(int msaa) const
{
  (void)msaa;
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
  VkResult     result = vkCreateRenderPass(m_device, &rpInfo, nullptr, &rp);
  assert(result == VK_SUCCESS);
  return rp;
}


bool ResourcesVK::initFramebuffer(int winWidth, int winHeight, int supersample, bool vsync)
{
  VkResult result;

  m_fboChangeID++;

  if(m_framebuffer.imgColor != nullptr)
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

  LOGI("framebuffer: %d x %d (%d msaa)\n", m_framebuffer.renderWidth, m_framebuffer.renderHeight, m_framebuffer.msaa)

  m_framebuffer.useResolved = supersample > 1;

  if(oldMsaa != m_framebuffer.msaa || oldResolved != m_framebuffer.useResolved)
  {
    vkDestroyRenderPass(m_device, m_framebuffer.passClear, nullptr);
    vkDestroyRenderPass(m_device, m_framebuffer.passPreserve, nullptr);
    vkDestroyRenderPass(m_device, m_framebuffer.passUI, nullptr);

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
  result                = vkCreateImageView(m_device, &cbImageViewInfo, nullptr, &m_framebuffer.viewColor);
  assert(result == VK_SUCCESS);

  if(m_framebuffer.useResolved)
  {
    cbImageViewInfo.image = m_framebuffer.imgColorResolved;
    result                = vkCreateImageView(m_device, &cbImageViewInfo, nullptr, &m_framebuffer.viewColorResolved);
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
  result                = vkCreateImageView(m_device, &dsImageViewInfo, nullptr, &m_framebuffer.viewDepthStencil);
  assert(result == VK_SUCCESS);
  // initial resource transitions
  {
    VkCommandBuffer cmd = createTempCmdBuffer();

    m_swapChain->cmdUpdateBarriers(cmd);

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
    result            = vkCreateFramebuffer(m_device, &fbInfo, nullptr, &fb);
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
    result            = vkCreateFramebuffer(m_device, &fbInfo, nullptr, &fb);
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
    attributes[2 + i].location = VERTEX_EXTRAS + i;
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

  // create a color blend attachment that does the blending
  VkPipelineColorBlendAttachmentState cbAttachmentState[1] = {};
  cbAttachmentState[0].blendEnable                         = VK_FALSE;
  cbAttachmentState[0].colorWriteMask =
      VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

  // create a color blend state that does the blending
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

  VkGraphicsPipelineCreateInfo pipelineInfo = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
  pipelineInfo.pVertexInputState            = &viStateInfo;
  pipelineInfo.pInputAssemblyState          = &iaStateInfo;
  pipelineInfo.pViewportState               = &vpStateInfo;
  pipelineInfo.pColorBlendState             = &cbStateInfo;
  pipelineInfo.pDepthStencilState           = &dsStateInfo;
  pipelineInfo.pMultisampleState            = &msStateInfo;
  pipelineInfo.pTessellationState           = &tessStateInfo;
  pipelineInfo.pDynamicState                = &dynStateInfo;

  pipelineInfo.renderPass = m_framebuffer.passPreserve;
  pipelineInfo.subpass    = 0;

  VkPipelineShaderStageCreateInfo stages[3];
  memset(stages, 0, sizeof(stages));
  pipelineInfo.pStages = stages;

  stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages[0].pName = "main";
  stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages[1].pName = "main";
  stages[2].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages[2].pName = "main";


  {
    pipelineInfo.pRasterizationState = &rsStateInfo;
    pipelineInfo.layout              = m_setupStandard.container.getPipeLayout();
    iaStateInfo.topology             = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    pipelineInfo.stageCount = 2;
    stages[0].stage         = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module        = m_shaderManager.get(m_shaders.standard_vertex);
    stages[1].stage         = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module        = m_shaderManager.get(m_shaders.standard_fragment);

    result = vkCreateGraphicsPipelines(m_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_setupStandard.pipeline);
    assert(result == VK_SUCCESS);
  }

  {
    pipelineInfo.pRasterizationState            = &rsStateInfoBbox;
    pipelineInfo.layout                         = m_setupBbox.container.getPipeLayout();
    iaStateInfo.topology                        = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
    viStateInfo.vertexBindingDescriptionCount   = 0;
    viStateInfo.vertexAttributeDescriptionCount = 0;
    viStateInfo.pVertexAttributeDescriptions    = nullptr;
    viStateInfo.pVertexBindingDescriptions      = nullptr;

    pipelineInfo.stageCount = 3;
    stages[0].stage         = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module        = m_shaderManager.get(m_shaders.bbox_vertex);
    stages[1].stage         = VK_SHADER_STAGE_GEOMETRY_BIT;
    stages[1].module        = m_shaderManager.get(m_shaders.bbox_geometry);
    stages[2].stage         = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[2].module        = m_shaderManager.get(m_shaders.bbox_fragment);

    result = vkCreateGraphicsPipelines(m_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_setupBbox.pipeline);
    assert(result == VK_SUCCESS);
  }

  // enable manually for debugging etc.
  bool dumpPipeInternals = false && m_context->hasDeviceExtension(VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME);
  
  // ensures the assumption in `Sample::getShaderPrepend()` that this value is used for mesh-shaders is actually true
  VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT rss_info = {
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT};
  rss_info.requiredSubgroupSize = m_subgroupSize;

  VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT* rss_info_ptr = nullptr;
  if(m_context->hasDeviceExtension(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME))
  {
    rss_info_ptr = &rss_info;
  }

  for(uint32_t isNV = 0; isNV < 2; isNV++)
  {
    if((isNV && !m_supportsMeshNV) || (!isNV && !m_supportsMeshEXT))
      continue;

    DrawSetup&           setup   = isNV ? m_setupMeshNV : m_setupMeshEXT;
    MeshShaderModuleIDs& shaders = isNV ? m_shaders.meshNV : m_shaders.meshEXT;

    // keep viStateInfo like above, i.e. no vertex inputs
    pipelineInfo.flags = dumpPipeInternals ? VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR : 0;
    pipelineInfo.pRasterizationState = &rsStateInfo;
    pipelineInfo.pVertexInputState   = nullptr;
    pipelineInfo.pInputAssemblyState = nullptr;
    pipelineInfo.layout              = setup.container.getPipeLayout();

    {
      pipelineInfo.stageCount = 2;
      stages[0].stage         = VK_SHADER_STAGE_MESH_BIT_NV;
      stages[0].module        = m_shaderManager.get(shaders.mesh);
      stages[0].pNext         = rss_info_ptr;
      stages[1].stage         = VK_SHADER_STAGE_FRAGMENT_BIT;
      stages[1].module        = m_shaderManager.get(shaders.mesh_fragment);
      stages[1].pNext         = nullptr;

      result = vkCreateGraphicsPipelines(m_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &setup.pipeline);
      assert(result == VK_SUCCESS);

      if(dumpPipeInternals)
      {
        nvvk::dumpPipelineInternals(m_device, setup.pipeline, isNV ? "pipeinternals_mesh_nv" : "pipeinternals_mesh_ext");
      }
    }

    {
      pipelineInfo.stageCount = 3;
      stages[0].stage         = VK_SHADER_STAGE_TASK_BIT_NV;
      stages[0].module        = m_shaderManager.get(shaders.task);
      stages[0].pNext         = rss_info_ptr;
      stages[1].stage         = VK_SHADER_STAGE_MESH_BIT_NV;
      stages[1].module        = m_shaderManager.get(shaders.task_mesh);
      stages[1].pNext         = rss_info_ptr;
      stages[2].stage         = VK_SHADER_STAGE_FRAGMENT_BIT;
      stages[2].module        = m_shaderManager.get(shaders.mesh_fragment);
      stages[2].pNext         = nullptr;

      result = vkCreateGraphicsPipelines(m_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &setup.pipelineTask);
      assert(result == VK_SUCCESS);


      if(dumpPipeInternals)
      {
        nvvk::dumpPipelineInternals(m_device, setup.pipelineTask, isNV ? "pipeinternals_taskmesh_nv" : "pipeinternals_taskmesh_ext");
      }
    }

    {
      pipelineInfo.stageCount = 2;
      stages[0].stage         = VK_SHADER_STAGE_MESH_BIT_NV;
      stages[0].module        = m_shaderManager.get(shaders.cull_mesh);
      stages[0].pNext         = rss_info_ptr;
      stages[1].stage         = VK_SHADER_STAGE_FRAGMENT_BIT;
      stages[1].module        = m_shaderManager.get(shaders.mesh_fragment);
      stages[1].pNext         = nullptr;

      result = vkCreateGraphicsPipelines(m_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &setup.pipelineCull);
      assert(result == VK_SUCCESS);
    }

    {
      pipelineInfo.stageCount = 3;
      stages[0].stage         = VK_SHADER_STAGE_TASK_BIT_NV;
      stages[0].module        = m_shaderManager.get(shaders.task);
      stages[0].pNext         = rss_info_ptr;
      stages[1].stage         = VK_SHADER_STAGE_MESH_BIT_NV;
      stages[1].module        = m_shaderManager.get(shaders.cull_task_mesh);
      stages[1].pNext         = rss_info_ptr;
      stages[2].stage         = VK_SHADER_STAGE_FRAGMENT_BIT;
      stages[2].module        = m_shaderManager.get(shaders.mesh_fragment);
      stages[2].pNext         = nullptr;

      result = vkCreateGraphicsPipelines(m_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &setup.pipelineCullTask);
      assert(result == VK_SUCCESS);
    }
  }
}

void ResourcesVK::deinitPipes()
{
  vkDestroyPipeline(m_device, m_setupStandard.pipeline, nullptr);
  m_setupStandard.pipeline = nullptr;
  vkDestroyPipeline(m_device, m_setupBbox.pipeline, nullptr);
  m_setupBbox.pipeline = nullptr;

  for(uint32_t isNV = 0; isNV < 2; isNV++)
  {
    if((isNV && !m_supportsMeshNV) || (!isNV && !m_supportsMeshEXT))
      continue;

    DrawSetup& setup = isNV ? m_setupMeshNV : m_setupMeshEXT;

    vkDestroyPipeline(m_device, setup.pipeline, nullptr);
    setup.pipeline = nullptr;
    vkDestroyPipeline(m_device, setup.pipelineTask, nullptr);
    setup.pipelineTask = nullptr;

    vkDestroyPipeline(m_device, setup.pipelineCull, nullptr);
    setup.pipelineCull = nullptr;
    vkDestroyPipeline(m_device, setup.pipelineCullTask, nullptr);
    setup.pipelineCullTask = nullptr;
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
                         0, nullptr, 0, nullptr, 1, &memBarrier);
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
    memBarrier.srcAccessMask        = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    memBarrier.dstAccessMask        = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    memBarrier.oldLayout            = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    memBarrier.newLayout            = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    memBarrier.image                = m_framebuffer.imgDepthStencil;
    memBarrier.subresourceRange     = depthStencilRange;

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
                         VK_FALSE, 0, nullptr, 0, nullptr, 1, &memBarrier);
  }
}


void ResourcesVK::cmdImageTransition(VkCommandBuffer    cmd,
                                     VkImage            img,
                                     VkImageAspectFlags aspects,
                                     VkAccessFlags      src,
                                     VkAccessFlags      dst,
                                     VkImageLayout      oldLayout,
                                     VkImageLayout      newLayout)
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

  vkCmdPipelineBarrier(cmd, srcPipe, dstPipe, VK_FALSE, 0, nullptr, 0, nullptr, 1, &memBarrier);
}

VkCommandBuffer ResourcesVK::createCmdBuffer(VkCommandPool pool, bool singleshot, bool primary, bool secondaryInClear) const
{
  VkResult result;

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

  {
    // Allocation phase

    {
      m_setupStandard.container.at(DSET_SCENE).initPool(1);
      m_setupStandard.container.at(DSET_OBJECT).initPool(1);


      m_setupBbox.container.at(DSET_SCENE).initPool(1);
      m_setupBbox.container.at(DSET_OBJECT).initPool(1);
      m_setupBbox.container.at(DSET_GEOMETRY).initPool(uint32_t(m_scene.m_geometryMem.getChunkCount()));
    }
    for(uint32_t isNV = 0; isNV < 2; isNV++)
    {
      if((isNV && !m_supportsMeshNV) || (!isNV && !m_supportsMeshEXT))
        continue;

      DrawSetup& setup = isNV ? m_setupMeshNV : m_setupMeshEXT;
      setup.container.at(DSET_SCENE).initPool(1);
      setup.container.at(DSET_OBJECT).initPool(1);
      setup.container.at(DSET_GEOMETRY).initPool(uint32_t(m_scene.m_geometryMem.getChunkCount()));
    }
  }

  {
    // Fill phase
    {
      {
        VkWriteDescriptorSet updateDescriptors[] = {
            m_setupStandard.container.at(DSET_SCENE).makeWrite(0, SCENE_UBO_VIEW, &m_common.viewInfo),
            m_setupStandard.container.at(DSET_OBJECT).makeWrite(0, 0, &m_scene.m_infos.matricesSingle),
            m_setupBbox.container.at(DSET_SCENE).makeWrite(0, SCENE_UBO_VIEW, &m_common.viewInfo),
            m_setupBbox.container.at(DSET_OBJECT).makeWrite(0, 0, &m_scene.m_infos.matricesSingle),

        };
        vkUpdateDescriptorSets(m_device, NV_ARRAY_SIZE(updateDescriptors), updateDescriptors, 0, nullptr);
      }

      for(uint32_t isNV = 0; isNV < 2; isNV++)
      {
        if((isNV && !m_supportsMeshNV) || (!isNV && !m_supportsMeshEXT))
          continue;

        DrawSetup&           setup               = isNV ? m_setupMeshNV : m_setupMeshEXT;
        VkWriteDescriptorSet updateDescriptors[] = {
            setup.container.at(DSET_SCENE).makeWrite(0, SCENE_UBO_VIEW, &m_common.viewInfo),
            setup.container.at(DSET_SCENE).makeWrite(0, SCENE_SSBO_STATS, &m_common.statsInfo),
            setup.container.at(DSET_OBJECT).makeWrite(0, 0, &m_scene.m_infos.matricesSingle),
        };
        vkUpdateDescriptorSets(m_device, NV_ARRAY_SIZE(updateDescriptors), updateDescriptors, 0, nullptr);
      }
    }
    {
      std::vector<VkWriteDescriptorSet> writeUpdates;

      for(VkDeviceSize g = 0; g < m_scene.m_geometryMem.getChunkCount(); g++)
      {
        const auto& chunk = m_scene.m_geometryMem.getChunk(g);

        writeUpdates.push_back(m_setupBbox.container.at(DSET_GEOMETRY).makeWrite(g, GEOMETRY_SSBO_MESHLETDESC, &chunk.meshInfo));

        writeUpdates.push_back(m_setupBbox.container.at(DSET_GEOMETRY).makeWrite(g, GEOMETRY_SSBO_PRIM, &chunk.meshIndicesInfo));
        writeUpdates.push_back(m_setupBbox.container.at(DSET_GEOMETRY).makeWrite(g, GEOMETRY_TEX_VBO, &chunk.vboView));
        writeUpdates.push_back(m_setupBbox.container.at(DSET_GEOMETRY).makeWrite(g, GEOMETRY_TEX_ABO, &chunk.aboView));

        for(uint32_t isNV = 0; isNV < 2; isNV++)
        {
          if((isNV && !m_supportsMeshNV) || (!isNV && !m_supportsMeshEXT))
            continue;

          DrawSetup& setup = isNV ? m_setupMeshNV : m_setupMeshEXT;

          writeUpdates.push_back(setup.container.at(DSET_GEOMETRY).makeWrite(g, GEOMETRY_SSBO_MESHLETDESC, &chunk.meshInfo));

          writeUpdates.push_back(setup.container.at(DSET_GEOMETRY).makeWrite(g, GEOMETRY_SSBO_PRIM, &chunk.meshIndicesInfo));
          writeUpdates.push_back(setup.container.at(DSET_GEOMETRY).makeWrite(g, GEOMETRY_TEX_VBO, &chunk.vboView));
          writeUpdates.push_back(setup.container.at(DSET_GEOMETRY).makeWrite(g, GEOMETRY_TEX_ABO, &chunk.aboView));
        }
      }

      vkUpdateDescriptorSets(m_device, (uint32_t)writeUpdates.size(), writeUpdates.data(), 0, nullptr);
    }
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

  m_setupStandard.container.deinitPools();
  m_setupBbox.container.deinitPools();

  for(uint32_t isNV = 0; isNV < 2; isNV++)
  {
    if((isNV && !m_supportsMeshNV) || (!isNV && !m_supportsMeshEXT))
      continue;

    DrawSetup& setup = isNV ? m_setupMeshNV : m_setupMeshEXT;
    setup.container.deinitPools();
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
  const RenderList::DrawItem* NV_RESTRICT drawItems = list->m_drawItems.data();
  size_t                                  numItems  = list->m_drawItems.size();

  const ResourcesVK* NV_RESTRICT res = this;

  const ResourcesVK::DrawSetup& setup = res->m_setupBbox;

  VkCommandBuffer cmd = res->createCmdBuffer(pool, false, false, true);
  res->cmdDynamicState(cmd);

  int lastGeometry = -1;
  int lastMatrix   = -1;
  int lastChunk    = -1;

  bool first = true;
  for(unsigned int i = 0; i < numItems; i++)
  {
    const RenderList::DrawItem& di = drawItems[i];

    if(first)
    {
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, setup.pipeline);

      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, setup.container.getPipeLayout(), DSET_SCENE, 1,
                              setup.container.at(DSET_SCENE).getSets(), 0, nullptr);

      first = false;
    }

    if(lastGeometry != di.geometryIndex)
    {
      const CadSceneVK::Geometry& geovk = m_scene.m_geometry[di.geometryIndex];
      int                         chunk = int(geovk.allocation.chunkIndex);


      if(chunk != lastChunk)
      {
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, setup.container.getPipeLayout(), DSET_GEOMETRY, 1,
                                setup.container.at(DSET_GEOMETRY).getSets() + chunk, 0, nullptr);

        lastChunk = chunk;
      }

      uint32_t offsets[4] = {uint32_t(geovk.meshletDesc.offset / sizeof(NVMeshlet::MeshletDesc)), 0, 0, 0};
      vkCmdPushConstants(cmd, setup.container.getPipeLayout(), VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(offsets), offsets);

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
