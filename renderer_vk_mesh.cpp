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


#include "nvmeshlet_builder.hpp"
#include "renderer.hpp"
#include "resources_vk.hpp"
#include <algorithm>
#include <assert.h>

#include <nvh/nvprint.hpp>
#include <nvmath/nvmath_glsltypes.h>

#include "common.h"


namespace meshlettest {

//////////////////////////////////////////////////////////////////////////


class RendererMeshVK : public Renderer
{
public:
  class TypeCmd : public Renderer::Type
  {
    bool        isAvailable(const nvvk::Context* context) const { return ResourcesVK::isAvailable() && context->hasDeviceExtension(VK_NV_MESH_SHADER_EXTENSION_NAME); }
    const char* name() const { return "VK mesh"; }
    Renderer*   create() const
    {
      RendererMeshVK* renderer = new RendererMeshVK();
      return renderer;
    }
    unsigned int priority() const { return 14; }

    Resources* resources() { return ResourcesVK::get(); }
  };


public:
  bool init(RenderList* NV_RESTRICT list, Resources* resources, const Config& config) override;
  void deinit() override;
  void draw(const FrameConfig& global) override;


  RendererMeshVK() {}

private:
  const RenderList* NV_RESTRICT m_list;
  ResourcesVK* NV_RESTRICT m_resources;
  Config                   m_config;

  VkCommandPool   m_cmdPool;
  VkCommandBuffer m_cmdBuffers[2];  // scene + bboxes
  size_t          m_fboChangeID;
  size_t          m_pipeChangeID;

  void GenerateCmdBuffers()
  {
    const RenderList::DrawItem* NV_RESTRICT drawItems           = m_list->m_drawItems.data();
    size_t                                  numItems            = m_list->m_drawItems.size();
    size_t                                  vertexSize          = m_list->m_scene->getVertexSize();
    size_t                                  vertexAttributeSize = m_list->m_scene->getVertexAttributeSize();

    const ResourcesVK* NV_RESTRICT res     = m_resources;
    const CadSceneVK&              sceneVK = res->m_scene;

    const ResourcesVK::DrawSetup& setup = res->m_setupMeshTask;

    VkCommandBuffer cmd = res->createCmdBuffer(m_cmdPool, false, false, true);
    res->cmdDynamicState(cmd);

    int  lastMaterial = -1;
    int  lastGeometry = -1;
    int  lastMatrix   = -1;
    int  lastChunk    = -1;

    bool lastTask     = true;

    VkPipeline pipeline = m_config.useCulling ? setup.pipelineCull : setup.pipeline;
    VkPipeline pipelineTask = m_config.useCulling ? setup.pipelineCullTask : setup.pipelineTask;

    uint32_t psoStats = 0;

    bool first = true;
    for(size_t i = 0; i < numItems; i++)
    {
      const RenderList::DrawItem& di = drawItems[i];

      bool useTask = di.task;
      if(first || useTask != lastTask)
      {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, useTask ? pipelineTask : pipeline);

        if(first)
        {
          vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, setup.container.getPipeLayout(), DSET_SCENE, 1,
                                  setup.container.at(DSET_SCENE).getSets(), 0, nullptr);
        }

        first    = false;
        lastTask = useTask;

        psoStats++;
      }

      if(lastGeometry != di.geometryIndex)
      {
        const CadSceneVK::Geometry& geo   = sceneVK.m_geometry[di.geometryIndex];
        int                         chunk = int(geo.allocation.chunkIndex);

        if(chunk != lastChunk)
        {
          vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, setup.container.getPipeLayout(), DSET_GEOMETRY,
                                  1, setup.container.at(DSET_GEOMETRY).getSets() + chunk, 0, nullptr);

          lastChunk  = chunk;
        }

        // we use the same vertex offset for both vbo and abo, our allocator should ensure this condition.
        assert(uint32_t(geo.vbo.offset / vertexSize) == uint32_t(geo.abo.offset / vertexAttributeSize));

        uint32_t offsets[4] = {uint32_t(geo.meshletDesc.offset / sizeof(NVMeshlet::MeshletDesc)),
                               uint32_t(geo.meshletPrim.offset), 0, uint32_t(geo.vbo.offset / vertexSize)};

        vkCmdPushConstants(cmd, setup.container.getPipeLayout(),
                           VK_SHADER_STAGE_TASK_BIT_NV | VK_SHADER_STAGE_MESH_BIT_NV, 0, sizeof(offsets), offsets);

        lastGeometry = di.geometryIndex;
      }

      if(lastMatrix != di.matrixIndex)
      {
        uint32_t offset = di.matrixIndex * res->m_alignedMatrixSize;
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, setup.container.getPipeLayout(), DSET_OBJECT, 1,
                                setup.container.at(DSET_OBJECT).getSets(), 1, &offset);
        lastMatrix = di.matrixIndex;
      }

      if(useTask)
      {
        nvmath::uvec4 assigns;
        assigns.x = di.meshlet.offset;
        assigns.y = di.meshlet.offset + di.meshlet.count - 1;
        assigns.z = 0;
        assigns.w = 0;
        vkCmdPushConstants(cmd, setup.container.getPipeLayout(), VK_SHADER_STAGE_TASK_BIT_NV,
                           sizeof(uint32_t) * 4, sizeof(assigns), &assigns);
      }

      uint32_t count  = useTask ? ((di.meshlet.count + m_list->m_config.taskNumMeshlets - 1) / m_list->m_config.taskNumMeshlets) : di.meshlet.count;
      uint32_t offset = useTask ? 0 : di.meshlet.offset;
      vkCmdDrawMeshTasksNV(cmd, count, offset);
    }

    vkEndCommandBuffer(cmd);

    m_cmdBuffers[0] = cmd;
    m_cmdBuffers[1] = res->createBoundingBoxCmdBuffer(m_cmdPool, m_list);

    m_fboChangeID  = res->m_fboChangeID;
    m_pipeChangeID = res->m_pipeChangeID;

    LOGI("cmdbuffer pso binds: %d\n", psoStats);
  }

  void DeleteCmdbuffers()
  {
    vkFreeCommandBuffers(m_resources->m_device, m_cmdPool, NV_ARRAY_SIZE(m_cmdBuffers), m_cmdBuffers);
  }
};


static RendererMeshVK::TypeCmd s_type_cmdbuffer_vk;

bool RendererMeshVK::init(RenderList* NV_RESTRICT list, Resources* resources, const Config& config)
{
  m_list      = list;
  m_resources = (ResourcesVK*)resources;
  m_config    = config;

  VkResult                result;
  VkCommandPoolCreateInfo cmdPoolInfo = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  cmdPoolInfo.queueFamilyIndex        = 0;
  result                              = vkCreateCommandPool(m_resources->m_device, &cmdPoolInfo, nullptr, &m_cmdPool);
  assert(result == VK_SUCCESS);

  GenerateCmdBuffers();

  return true;
}

void RendererMeshVK::deinit()
{
  DeleteCmdbuffers();
  vkDestroyCommandPool(m_resources->m_device, m_cmdPool, nullptr);
}

void RendererMeshVK::draw(const FrameConfig& global)
{
  ResourcesVK* NV_RESTRICT res = m_resources;

  if(m_pipeChangeID != res->m_pipeChangeID || m_fboChangeID != res->m_fboChangeID)
  {
    DeleteCmdbuffers();
    GenerateCmdBuffers();
  }

  VkCommandBuffer primary = res->createTempCmdBuffer();

  {
    const nvvk::ProfilerVK::Section profile(res->m_profilerVK, "Render", primary);

    vkCmdUpdateBuffer(primary, res->m_common.viewBuffer, 0, sizeof(SceneData), (const uint32_t*)&global.sceneUbo);
    vkCmdUpdateBuffer(primary, res->m_common.statsBuffer, 0, sizeof(CullStats), (const uint32_t*)&m_list->m_stats);

    res->cmdPipelineBarrier(primary);
    // clear via pass
    res->cmdBeginRenderPass(primary, true, true);
    vkCmdExecuteCommands(primary, global.meshletBoxes ? 2 : 1, m_cmdBuffers);
    vkCmdEndRenderPass(primary);
    res->cmdCopyStats(primary);
  }

  vkEndCommandBuffer(primary);
  res->submissionEnqueue(primary);
}


}  // namespace meshlettest
