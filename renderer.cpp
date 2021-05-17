/*
 * Copyright (c) 2016-2021, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2016-2021 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
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
