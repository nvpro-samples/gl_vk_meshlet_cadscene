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

#include "renderer.hpp"
#include "resources_vk.hpp"
#include <algorithm>
#include <assert.h>

#include <nvmath/nvmath_glsltypes.h>

#include "common.h"


namespace meshlettest {

//////////////////////////////////////////////////////////////////////////


class RendererVK : public Renderer
{
public:
  class TypeCmd : public Renderer::Type
  {
    bool        isAvailable() const { return ResourcesVK::isAvailable(); }
    const char* name() const { return "VK standard"; }
    Renderer*   create() const
    {
      RendererVK* renderer = new RendererVK();
      return renderer;
    }
    unsigned int priority() const { return 10; }

    Resources* resources() { return ResourcesVK::get(); }
  };


public:
  bool init(RenderList* NV_RESTRICT list, Resources* resources, const Config& config) override;
  void deinit() override;
  void draw(const FrameConfig& global) override;


  RendererVK() {}

private:
  const RenderList* NV_RESTRICT m_list;
  ResourcesVK* NV_RESTRICT m_resources;
  Config                   m_config;

  VkCommandPool   m_cmdPool;
  VkCommandBuffer m_cmdBuffers[2];
  size_t          m_fboChangeID;
  size_t          m_pipeChangeID;

  void GenerateCmdBuffers()
  {
    const RenderList::DrawItem* NV_RESTRICT drawItems = m_list->m_drawItems.data();
    size_t                                  numItems  = m_list->m_drawItems.size();

    const ResourcesVK* NV_RESTRICT res     = m_resources;
    const CadSceneVK&              sceneVK = res->m_scene;

    const ResourcesVK::DrawSetup& setup = res->m_setupRegular;

    VkCommandBuffer cmd = res->createCmdBuffer(m_cmdPool, false, false, true);
    res->cmdDynamicState(cmd);

    int lastMaterial = -1;
    int lastGeometry = -1;
    int lastMatrix   = -1;
    int lastChunk    = -1;

    bool first = true;
    for(size_t i = 0; i < numItems; i++)
    {
      const RenderList::DrawItem& di = drawItems[i];

      if(first)
      {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, setup.pipeline);

        if(first)
        {
          vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, setup.container.getPipeLayout(), DSET_SCENE, 1,
                                  setup.container.at(DSET_SCENE).getSets(), 0, NULL);
        }

        first = false;
      }

      if(lastGeometry != di.geometryIndex)
      {
        const CadSceneVK::Geometry& geo = sceneVK.m_geometry[di.geometryIndex];

        vkCmdBindVertexBuffers(cmd, 0, 1, &geo.vbo.buffer, &geo.vbo.offset);
        vkCmdBindVertexBuffers(cmd, 1, 1, &geo.abo.buffer, &geo.abo.offset);
        vkCmdBindIndexBuffer(cmd, geo.ibo.buffer, geo.ibo.offset, di.shorts ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32);

        lastGeometry = di.geometryIndex;
      }

      if(lastMatrix != di.matrixIndex)
      {
        uint32_t offset = di.matrixIndex * res->m_alignedMatrixSize;
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, setup.container.getPipeLayout(), DSET_OBJECT, 1,
                                setup.container.at(DSET_OBJECT).getSets(), 1, &offset);
        lastMatrix = di.matrixIndex;
      }

      // drawcall
      size_t indexSize = di.shorts ? sizeof(uint16_t) : sizeof(uint32_t);
      vkCmdDrawIndexed(cmd, di.range.count, 1, uint32_t(di.range.offset / indexSize), 0, 0);
    }

    vkEndCommandBuffer(cmd);

    m_cmdBuffers[0] = cmd;
    m_cmdBuffers[1] = res->createBoundingBoxCmdBuffer(m_cmdPool, m_list);

    m_fboChangeID  = res->m_fboChangeID;
    m_pipeChangeID = res->m_pipeChangeID;
  }

  void DeleteCmdbuffers()
  {
    vkFreeCommandBuffers(m_resources->m_device, m_cmdPool, NV_ARRAY_SIZE(m_cmdBuffers), m_cmdBuffers);
  }
};


static RendererVK::TypeCmd s_type_cmdbuffer_vk;

bool RendererVK::init(RenderList* NV_RESTRICT list, Resources* resources, const Config& config)
{
  m_list      = list;
  m_resources = (ResourcesVK*)resources;
  m_config    = config;

  VkResult                result;
  VkCommandPoolCreateInfo cmdPoolInfo = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  cmdPoolInfo.queueFamilyIndex        = 0;
  result                              = vkCreateCommandPool(m_resources->m_device, &cmdPoolInfo, NULL, &m_cmdPool);
  assert(result == VK_SUCCESS);

  GenerateCmdBuffers();

  return true;
}

void RendererVK::deinit()
{
  DeleteCmdbuffers();
  vkDestroyCommandPool(m_resources->m_device, m_cmdPool, NULL);
}

void RendererVK::draw(const FrameConfig& global)
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
