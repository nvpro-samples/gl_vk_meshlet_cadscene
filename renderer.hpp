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



#ifndef RENDERER_H__
#define RENDERER_H__

#include "resources.hpp"
#include <nvh/profiler.hpp>

namespace meshlettest {

class RenderList
{
public:
  enum Strategy
  {                   
    STRATEGY_SINGLE,     // entire geometry
    STRATEGY_INDIVIDUAL, // per-object
  };

  struct Config
  {
    Strategy strategy;
    uint32_t objectFrom;
    uint32_t objectNum;
    int32_t  indexThreshold;
    uint32_t taskMinMeshlets;
    uint32_t taskNumMeshlets = 32;
    uint32_t meshNumMeshlets = 1;
  };

  struct DrawItem
  {
    bool                   task;
    bool                   shorts;
    int                    geometryIndex;
    int                    matrixIndex;
    int                    cullIndex;
    CadScene::DrawRange    range;
    CadScene::MeshletRange meshlet;
  };

  void setup(const CadScene* NV_RESTRICT scene, const Config& config);

  CullStats       m_stats;
  Config          m_config;
  const CadScene* NV_RESTRICT m_scene;
  std::vector<DrawItem>       m_drawItems;
};

class Renderer
{
public:
  struct Config
  {
    bool useCulling = false;
  };

  class Type
  {
  public:
    Type() { getRegistry().push_back(this); }

  public:
#if IS_OPENGL
    virtual bool         isAvailable(const nvgl::ContextWindow* contextWindow) const = 0;
#elif IS_VULKAN
    virtual bool         isAvailable(const nvvk::Context* context) const = 0;
#endif
    virtual const char*  name() const        = 0;
    virtual Renderer*    create() const      = 0;
    virtual unsigned int priority() const { return 0xFF; }

    virtual Resources* resources() = 0;
  };

  typedef std::vector<Type*> Registry;

  static Registry& getRegistry()
  {
    static Registry s_registry;
    return s_registry;
  }

public:
  virtual bool init(RenderList* NV_RESTRICT list, Resources* resources, const Config& config) = 0;
  virtual void deinit()                                                                       = 0;
  virtual void draw(const FrameConfig& global)                                                = 0;

  virtual ~Renderer() {}
};
}  // namespace meshlettest

#endif
