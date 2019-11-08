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

#include <algorithm>
#include <assert.h>

#include <nvmath/nvmath_glsltypes.h>

#include "nvmeshlet_builder.hpp"
#include "renderer.hpp"
#include "resources_gl.hpp"

#include "common.h"

namespace meshlettest {

//////////////////////////////////////////////////////////////////////////

class RendererMeshGL : public Renderer
{
public:
  class Type : public Renderer::Type
  {
    bool        isAvailable() const { return has_GL_NV_mesh_shader != 0; }
    const char* name() const { return "GL mesh"; }
    Renderer*   create() const
    {
      RendererMeshGL* renderer = new RendererMeshGL();
      return renderer;
    }

    Resources* resources() { return ResourcesGL::get(); }

    unsigned int priority() const { return 4; }
  };
  class TypeVbum : public Renderer::Type
  {
    bool isAvailable() const
    {
      return has_GL_NV_vertex_buffer_unified_memory && has_GL_NV_uniform_buffer_unified_memory && has_GL_NV_mesh_shader;
    }
    const char* name() const { return "GL mesh nvbindless"; }
    Renderer*   create() const
    {
      RendererMeshGL* renderer = new RendererMeshGL();
      renderer->m_bindless     = true;
      return renderer;
    }
    unsigned int priority() const { return 4; }

    Resources* resources() { return ResourcesGL::get(); }
  };

public:
  bool init(RenderList* NV_RESTRICT list, Resources* resources, const Config& config) override;
  void deinit() override;
  void draw(const FrameConfig& global) override;

  bool m_bindless = false;

private:
  const RenderList* NV_RESTRICT m_list;
  ResourcesGL* NV_RESTRICT m_resources;
  Config                   m_config;
};

static RendererMeshGL::Type     s_uborange;
static RendererMeshGL::TypeVbum s_uborange_vbum;

bool RendererMeshGL::init(RenderList* NV_RESTRICT list, Resources* resources, const Config& config)
{
  m_list      = list;
  m_resources = (ResourcesGL*)resources;
  m_config    = config;

  return true;
}

void RendererMeshGL::deinit() {}

void RendererMeshGL::draw(const FrameConfig& global)
{
  ResourcesGL* NV_RESTRICT res        = m_resources;
  const CadScene* NV_RESTRICT scene   = m_list->m_scene;
  const CadSceneGL&           sceneGL = res->m_scene;


  const nvgl::ProfilerGL::Section profile(res->m_profilerGL, "Render");

  bool   bindless            = m_bindless;
  size_t vertexSize          = scene->getVertexSize();
  size_t vertexAttributeSize = scene->getVertexAttributeSize();

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

  glNamedBufferSubData(res->m_common.viewBuffer.buffer, 0, sizeof(SceneData), &global.sceneUbo);
  glNamedBufferSubData(res->m_common.statsBuffer.buffer, 0, sizeof(CullStats), &m_list->m_stats);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, SSBO_SCENE_STATS, res->m_common.statsBuffer);

  if(bindless)
  {
    glEnableClientState(GL_UNIFORM_BUFFER_UNIFIED_NV);

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

    bool lastShorts = false;
    bool lastTask   = false;

    int statsGeometry = 0;
    int statsMatrix   = 0;
    int statsMaterial = 0;
    int statsDraw     = 0;

    bool first = true;
    for(int i = 0; i < m_list->m_drawItems.size(); i++)
    {
      const RenderList::DrawItem& di = m_list->m_drawItems[i];

      bool useTask = di.task;

      if(first || useTask != lastTask)
      {
        glUseProgram(useTask ? res->m_programs.draw_object_mesh_task : res->m_programs.draw_object_mesh);
        lastTask = useTask;
        first    = false;
      }

      if(lastGeometry != di.geometryIndex)
      {
        const CadSceneGL::Geometry& geo   = sceneGL.m_geometry[di.geometryIndex];
        int                         chunk = int(geo.mem.chunkIndex);

#if USE_PER_GEOMETRY_VIEWS
        if(bindless)
        {
          glBufferAddressRangeNV(GL_UNIFORM_BUFFER_ADDRESS_NV, UBO_GEOMETRY,
                                 res->m_setup.geometryBindings.bufferADDR + sizeof(CadSceneGL::GeometryUbo) * di.geometryIndex,
                                 sizeof(CadSceneGL::GeometryUbo));
        }
        else
        {
          glBindBufferRange(GL_UNIFORM_BUFFER, UBO_GEOMETRY, res->m_setup.geometryBindings.buffer,
                            sizeof(CadSceneGL::GeometryUbo) * di.geometryIndex, sizeof(CadSceneGL::GeometryUbo));
        }
#else
        if(lastChunk != chunk || lastShorts != di.shorts)
        {
          int idx = chunk * 2 + (di.shorts ? 1 : 0);
          if(bindless)
          {
            glBufferAddressRangeNV(GL_UNIFORM_BUFFER_ADDRESS_NV, UBO_GEOMETRY,
                                   res->m_setup.geometryBindings.bufferADDR + sizeof(CadSceneGL::GeometryUbo) * idx,
                                   sizeof(CadSceneGL::GeometryUbo));
          }
          else
          {
            glBindBufferRange(GL_UNIFORM_BUFFER, UBO_GEOMETRY, res->m_setup.geometryBindings.buffer,
                              sizeof(CadSceneGL::GeometryUbo) * idx, sizeof(CadSceneGL::GeometryUbo));
          }

          lastChunk  = chunk;
          lastShorts = di.shorts;
        }

        // we use the same vertex offset for both vbo and abo, our allocator should ensure this condition.
        assert(uint32_t(geo.vbo.offset / vertexSize) == uint32_t(geo.abo.offset / vertexAttributeSize));

        glUniform4ui(0, uint32_t(geo.topoMeshlet.offset / sizeof(NVMeshlet::MeshletDesc)),
                     uint32_t(geo.topoPrim.offset / (NVMeshlet::PRIMITIVE_INDICES_PER_FETCH)),
                     uint32_t(geo.topoVert.offset / (di.shorts ? 2 : 4)), uint32_t(geo.vbo.offset / vertexSize));

#endif

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

      if(useTask)
      {
        glUniform4ui(1, di.meshlet.offset, di.meshlet.offset + di.meshlet.count - 1, 0, 0);
      }

      uint32_t offset = useTask ? 0 : di.meshlet.offset;
      uint32_t count  = useTask ? (di.meshlet.count + 31) / 32 : di.meshlet.count;

      glDrawMeshTasksNV(offset, count);

      statsDraw++;
    }

    statsGeometry;
    statsMatrix;
    statsDraw;
  }

  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, SSBO_SCENE_STATS, 0);
  glBindBufferBase(GL_UNIFORM_BUFFER, UBO_SCENE_VIEW, 0);
  glBindBufferBase(GL_UNIFORM_BUFFER, UBO_OBJECT, 0);
  glBindBufferBase(GL_UNIFORM_BUFFER, UBO_GEOMETRY, 0);

  res->copyStats();

  if(res->m_clipping)
  {
    for(int i = 0; i < NUM_CLIPPING_PLANES; i++)
    {
      glDisable(GL_CLIP_DISTANCE0 + i);
    }
  }

  if(m_bindless)
  {
    glDisableClientState(GL_UNIFORM_BUFFER_UNIFIED_NV);
  }

  if(global.meshletBoxes)
  {
    res->drawBoundingBoxes(m_list);
  }

  glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

}  // namespace meshlettest
