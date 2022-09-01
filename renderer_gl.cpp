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


#include "renderer.hpp"
#include "resources_gl.hpp"
#include <algorithm>

#include <nvmath/nvmath_glsltypes.h>

#include "common.h"

namespace meshlettest {

//////////////////////////////////////////////////////////////////////////


class RendererGL : public Renderer
{
public:
  class Type : public Renderer::Type
  {
    bool        isAvailable(const nvgl::ContextWindow* contextWindow) const override { return true; }
    [[nodiscard]] const char* name() const override { return "GL standard"; }
    [[nodiscard]] Renderer*   create() const override
    {
      auto* renderer = new RendererGL();
      return renderer;
    }

    Resources* resources() override { return ResourcesGL::get(); }

    [[nodiscard]] unsigned int priority() const override { return 0; }
  };
  class TypeVbum : public Renderer::Type
  {
    bool isAvailable(const nvgl::ContextWindow* contextWindow) const override
    {
      return has_GL_NV_vertex_buffer_unified_memory && has_GL_NV_uniform_buffer_unified_memory;
    }
    [[nodiscard]] const char* name() const override { return "GL standard nvbindless"; }
    [[nodiscard]] Renderer*   create() const override
    {
      auto* renderer = new RendererGL();
      renderer->m_bindless = true;
      return renderer;
    }
    [[nodiscard]] unsigned int priority() const override { return 0; }

    Resources* resources() override { return ResourcesGL::get(); }
  };

public:
  bool init(RenderList* NV_RESTRICT list, Resources* resources, const Config& config) override;
  void deinit() override;
  void draw(const FrameConfig& global) override;

  bool m_bindless = false;

private:
  const RenderList* NV_RESTRICT m_list{};
  ResourcesGL* NV_RESTRICT m_resources{};
  Config                   m_config;
};

static RendererGL::Type     s_uborange;
static RendererGL::TypeVbum s_uborange_vbum;

bool RendererGL::init(RenderList* NV_RESTRICT list, Resources* resources, const Config& config)
{
  m_list      = list;
  m_resources = (ResourcesGL*)resources;
  m_config    = config;
  return true;
}

void RendererGL::deinit() {}

void RendererGL::draw(const FrameConfig& global)
{
  ResourcesGL* NV_RESTRICT res     = m_resources;
  const CadSceneGL&        sceneGL = res->m_scene;

  const nvgl::ProfilerGL::Section profile(res->m_profilerGL, "Render");

  bool bindless = m_bindless;

  // generic state setup
  glViewport(0, 0, res->m_framebuffer.renderWidth, res->m_framebuffer.renderHeight);

  glBindFramebuffer(GL_FRAMEBUFFER, res->m_framebuffer.fboScene);
  glClearColor(0.2f, 0.2f, 0.2f, 0.0f);
  glClearDepth(1.0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

  glDepthFunc(GL_LESS);
  glEnable(GL_DEPTH_TEST);
  if(res->m_cullBackFace)
  {
    glEnable(GL_CULL_FACE);
  }
  else
  {
    glDisable(GL_CULL_FACE);
  }

  if(res->m_clipping)
  {
    for(int i = 0; i < NUM_CLIPPING_PLANES; i++)
    {
      glEnable(GL_CLIP_DISTANCE0 + i);
    }
  }

  glUseProgram(res->m_programs.draw_object_tris);

  glNamedBufferSubData(res->m_common.viewBuffer.buffer, 0, sizeof(SceneData), &global.sceneUbo);
  glNamedBufferSubData(res->m_common.statsBuffer.buffer, 0, sizeof(CullStats), &m_list->m_stats);

  res->enableVertexFormat();

  if(bindless)
  {
    glEnableClientState(GL_VERTEX_ATTRIB_ARRAY_UNIFIED_NV);
    glEnableClientState(GL_ELEMENT_ARRAY_UNIFIED_NV);
    glEnableClientState(GL_UNIFORM_BUFFER_UNIFIED_NV);

    glBufferAddressRangeNV(GL_VERTEX_ATTRIB_ARRAY_ADDRESS_NV, 0, 0, 0);
    glBufferAddressRangeNV(GL_ELEMENT_ARRAY_ADDRESS_NV, 0, 0, 0);

    glBufferAddressRangeNV(GL_UNIFORM_BUFFER_ADDRESS_NV, UBO_SCENE_VIEW, 0, 0);
    glBufferAddressRangeNV(GL_UNIFORM_BUFFER_ADDRESS_NV, UBO_OBJECT, 0, 0);
    glBufferAddressRangeNV(GL_UNIFORM_BUFFER_ADDRESS_NV, UBO_GEOMETRY, 0, 0);
  }

  if(bindless)
  {
    glBufferAddressRangeNV(GL_UNIFORM_BUFFER_ADDRESS_NV, UBO_SCENE_VIEW, res->m_common.viewBuffer.bufferADDR, sizeof(SceneData));
  }
  else
  {
    glBindBufferBase(GL_UNIFORM_BUFFER, UBO_SCENE_VIEW, res->m_common.viewBuffer.buffer);
  }

  {
    int lastMaterial = -1;
    int lastGeometry = -1;
    int lastMatrix   = -1;
    int lastChunk    = -1;

    int statsGeometry = 0;
    int statsMatrix   = 0;
    int statsMaterial = 0;
    int statsDraw     = 0;
    int statsChunk    = 0;

    for(const auto & di : m_list->m_drawItems)
    {
      const CadSceneGL::Geometry& geo = sceneGL.m_geometry[di.geometryIndex];

      GLintptr useOffset = bindless ? 0 : 1;

      if(lastGeometry != di.geometryIndex)
      {
        if(lastChunk != geo.mem.chunkIndex)
        {
          const auto& chunk = res->m_scene.m_geometryMem.getChunk(geo.mem);

          if(bindless)
          {
            glBufferAddressRangeNV(GL_VERTEX_ATTRIB_ARRAY_ADDRESS_NV, 0, chunk.vboADDR, static_cast<GLsizei>(chunk.vboSize));
            glBufferAddressRangeNV(GL_VERTEX_ATTRIB_ARRAY_ADDRESS_NV, 1, chunk.aboADDR, static_cast<GLsizei>(chunk.aboSize));
            glBufferAddressRangeNV(GL_ELEMENT_ARRAY_ADDRESS_NV, 0, chunk.iboADDR, static_cast<GLsizei>(chunk.iboSize));
          }
          else
          {
            glBindVertexBuffer(0, chunk.vboGL, 0, static_cast<GLsizei>(res->m_vertexSize));
            glBindVertexBuffer(1, chunk.aboGL, 0, static_cast<GLsizei>(res->m_vertexAttributeSize));
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, chunk.iboGL);
          }

          lastChunk = int(geo.mem.chunkIndex);

          statsChunk++;
        }

        lastGeometry = di.geometryIndex;

        statsGeometry++;
      }

      if(lastMatrix != di.matrixIndex)
      {

        if(bindless)
        {
          glBufferAddressRangeNV(GL_UNIFORM_BUFFER_ADDRESS_NV, UBO_OBJECT,
                                 res->m_scene.m_buffers.matrices.bufferADDR + res->m_alignedMatrixSize * di.matrixIndex,
                                 sizeof(CadScene::MatrixNode));
        }
        else
        {
          glBindBufferRange(GL_UNIFORM_BUFFER, UBO_OBJECT, res->m_scene.m_buffers.matrices.buffer,
                            res->m_alignedMatrixSize * di.matrixIndex, sizeof(CadScene::MatrixNode));
        }

        lastMatrix = di.matrixIndex;

        statsMatrix++;
      }

      glDrawElementsBaseVertex(GL_TRIANGLES, di.range.count, di.shorts ? GL_UNSIGNED_SHORT : GL_UNSIGNED_INT,
                               (void*)(di.range.offset + geo.ibo.offset), geo.vbo.offset / res->m_vertexSize);

      statsDraw++;
    }

    (void)statsGeometry;
    (void)statsMatrix;
    (void)statsDraw;
    (void)statsChunk;
  }

  glBindBufferBase(GL_UNIFORM_BUFFER, UBO_SCENE_VIEW, 0);
  glBindBufferBase(GL_UNIFORM_BUFFER, UBO_OBJECT, 0);
  glBindBufferBase(GL_UNIFORM_BUFFER, UBO_GEOMETRY, 0);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
  glBindVertexBuffer(0, 0, 0, 16);
  glBindVertexBuffer(1, 0, 0, 16);

  if(m_bindless)
  {
    glDisableClientState(GL_VERTEX_ATTRIB_ARRAY_UNIFIED_NV);
    glDisableClientState(GL_ELEMENT_ARRAY_UNIFIED_NV);
    glDisableClientState(GL_UNIFORM_BUFFER_UNIFIED_NV);
  }

  res->copyStats();

  res->disableVertexFormat();

  if(res->m_clipping)
  {
    for(int i = 0; i < NUM_CLIPPING_PLANES; i++)
    {
      glDisable(GL_CLIP_DISTANCE0 + i);
    }
  }

  if(global.meshletBoxes)
  {
    res->drawBoundingBoxes(m_list);
  }

  glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

}  // namespace meshlettest
