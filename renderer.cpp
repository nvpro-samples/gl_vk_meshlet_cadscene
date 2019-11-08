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
#include <algorithm>
#include <assert.h>

#include <nvh/nvprint.hpp>
#include <nvmath/nvmath_glsltypes.h>

#include "common.h"

#pragma pack(1)


namespace meshlettest {
bool     Resources::s_vkMeshSupport     = false;
bool     Resources::s_vkNVglslExtension = false;
uint32_t Resources::s_vkDevice          = 0;
uint32_t Resources::s_glDevice          = 0;

//////////////////////////////////////////////////////////////////////////

static void AddItem(std::vector<RenderList::DrawItem>& drawItems, const RenderList::Config& config, RenderList::DrawItem& di)
{
  bool passes = di.range.count > abs(config.indexThreshold);
  if(config.indexThreshold < 0)
  {
    passes = !passes;
  }
  di.task = config.minTaskMeshlets > 0 && di.meshlet.count >= config.minTaskMeshlets;
  if(di.range.count && passes)
  {
    drawItems.push_back(di);
  }
}

static void FillSingle(std::vector<RenderList::DrawItem>& drawItems,
                       const RenderList::Config&          config,
                       const CadScene::Object&            obj,
                       const CadScene::Geometry&          geo,
                       int                                objectIndex)
{
  if(!obj.parts[0].active || !geo.numIndexSolid)
    return;

  RenderList::DrawItem di;
  di.shorts         = geo.useShorts != 0;
  di.geometryIndex  = obj.geometryIndex;
  di.matrixIndex    = obj.matrixIndex;
  di.range.offset   = 0;
  di.range.count    = geo.numIndexSolid;
  di.meshlet.offset = 0;
  di.meshlet.count  = geo.meshlet.numMeshlets;

  AddItem(drawItems, config, di);
}

static void FillIndividual(std::vector<RenderList::DrawItem>& drawItems,
                           const RenderList::Config&          config,
                           const CadScene::Object&            obj,
                           const CadScene::Geometry&          geo,
                           int                                objectIndex)
{
  for(size_t p = 0; p < obj.parts.size(); p++)
  {
    const CadScene::ObjectPart&   part    = obj.parts[p];
    const CadScene::GeometryPart& partgeo = geo.parts[p];

    if(!part.active)
      continue;

    RenderList::DrawItem di;
    di.shorts        = geo.useShorts != 0;
    di.geometryIndex = obj.geometryIndex;
    di.matrixIndex   = part.matrixIndex;

    di.range   = partgeo.indexSolid;
    di.meshlet = partgeo.meshSolid;

    AddItem(drawItems, config, di);
  }
}

static inline bool DrawItem_compare_groups(const RenderList::DrawItem& a, const RenderList::DrawItem& b)
{
  int diff = 0;
  diff     = diff != 0 ? diff : ((a.task ? 1 : 0) - (b.task ? 1 : 0));
  diff     = diff != 0 ? diff : (a.geometryIndex - b.geometryIndex);
  diff     = diff != 0 ? diff : (a.matrixIndex - b.matrixIndex);

  return diff < 0;
}

void RenderList::setup(const CadScene* NV_RESTRICT scene, const Config& config)
{
  m_scene  = scene;
  m_config = config;
  m_drawItems.clear();

  size_t maxObjects = scene->m_objects.size();
  size_t from       = std::min(maxObjects - 1, size_t(config.objectFrom));
  maxObjects        = std::min(maxObjects, from + size_t(config.objectNum));

  for(size_t i = from; i < maxObjects; i++)
  {
    const CadScene::Object&   obj = scene->m_objects[i];
    const CadScene::Geometry& geo = scene->m_geometry[obj.geometryIndex];

    if(config.strategy == STRATEGY_SINGLE)
    {
      FillSingle(m_drawItems, config, obj, geo, int(i));
    }
    else if(config.strategy == STRATEGY_INDIVIDUAL)
    {
      FillIndividual(m_drawItems, config, obj, geo, int(i));
    }
  }

  memset(&m_stats, 0, sizeof(m_stats));

  uint32_t sumTriangles      = 0;
  uint32_t sumTrianglesShort = 0;
  for(size_t i = 0; i < m_drawItems.size(); i++)
  {
    const DrawItem& di = m_drawItems[i];
    sumTriangles += di.range.count / 3;
    sumTrianglesShort += (di.range.count / 3) * (di.shorts ? 1 : 0);
    m_stats.tasksInput += di.task ? (di.meshlet.count + 31) / 32 : 0;
    m_stats.meshletsInput += di.meshlet.count;
    m_stats.trisInput += di.range.count / 3;
  }
  LOGI("draw calls:      %9d\n", uint32_t(m_drawItems.size()));
  LOGI("triangles total: %9d\n", sumTriangles);
  LOGI("triangles short: %9d\n\n", sumTrianglesShort);

  std::sort(m_drawItems.begin(), m_drawItems.end(), DrawItem_compare_groups);
}
}  // namespace meshlettest
