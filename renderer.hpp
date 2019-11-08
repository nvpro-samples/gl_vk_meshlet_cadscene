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


#ifndef RENDERER_H__
#define RENDERER_H__

#include "resources.hpp"
#include <nvh/profiler.hpp>

namespace meshlettest {

class RenderList
{
public:
  enum Strategy
  {                   // per-object
    STRATEGY_SINGLE,  // entire geometry
    STRATEGY_INDIVIDUAL,
  };

  struct Config
  {
    Strategy strategy;
    uint32_t objectFrom;
    uint32_t objectNum;
    int32_t  indexThreshold;
    uint32_t minTaskMeshlets;
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
    bool blah = false;
  };

  class Type
  {
  public:
    Type() { getRegistry().push_back(this); }

  public:
    virtual bool         isAvailable() const = 0;
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
