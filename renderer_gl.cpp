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
#include "resources_gl.hpp"
#include <algorithm>
#include <assert.h>

#include <nvmath/nvmath_glsltypes.h>

#include "common.h"

namespace meshlettest {

//////////////////////////////////////////////////////////////////////////


class RendererGL : public Renderer
{
public:
  class Type : public Renderer::Type
  {
    bool        isAvailable() const { return true; }
    const char* name() const { return "GL standard"; }
    Renderer*   create() const
    {
      RendererGL* renderer = new RendererGL();
      return renderer;
    }

    Resources* resources() { return ResourcesGL::get(); }

    unsigned int priority() const { return 0; }
  };
  class TypeVbum : public Renderer::Type
  {
    bool isAvailable() const
    {
      return has_GL_NV_vertex_buffer_unified_memory && has_GL_NV_uniform_buffer_unified_memory;
    }
    const char* name() const { return "GL standard nvbindless"; }
    Renderer*   create() const
    {
      RendererGL* renderer = new RendererGL();
      renderer->m_bindless = true;
      return renderer;
    }
    unsigned int priority() const { return 0; }

    Resources* resources() { return ResourcesGL::get(); }
  };

public:
  bool init(RenderList* NV_RESTRICT scene, Resources* resources, const Config& config) override;
  void deinit() override;
  void draw(const FrameConfig& global) override;

  bool m_bindless = false;

private:
  const RenderList* NV_RESTRICT m_list;
  ResourcesGL* NV_RESTRICT m_resources;
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

    for(int i = 0; i < m_list->m_drawItems.size(); i++)
    {
      const RenderList::DrawItem& di  = m_list->m_drawItems[i];
      const CadSceneGL::Geometry& geo = sceneGL.m_geometry[di.geometryIndex];

      GLintptr useOffset = bindless ? 0 : 1;

      if(lastGeometry != di.geometryIndex)
      {
        if(lastChunk != geo.mem.chunkIndex)
        {
          const auto& chunk = res->m_scene.m_geometryMem.getChunk(geo.mem);

          if(bindless)
          {
            glBufferAddressRangeNV(GL_VERTEX_ATTRIB_ARRAY_ADDRESS_NV, 0, chunk.vboADDR, chunk.vboSize);
            glBufferAddressRangeNV(GL_VERTEX_ATTRIB_ARRAY_ADDRESS_NV, 1, chunk.aboADDR, chunk.aboSize);
            glBufferAddressRangeNV(GL_ELEMENT_ARRAY_ADDRESS_NV, 0, chunk.iboADDR, chunk.iboSize);
          }
          else
          {
            glBindVertexBuffer(0, chunk.vboGL, 0, res->m_vertexSize);
            glBindVertexBuffer(1, chunk.aboGL, 0, res->m_vertexAttributeSize);
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

    statsGeometry;
    statsMatrix;
    statsDraw;
    statsChunk;
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
