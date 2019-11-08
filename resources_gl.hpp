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

#pragma once

#include "resources.hpp"
#include <nvgl/base_gl.hpp>
#include <nvgl/profiler_gl.hpp>
#include <nvgl/programmanager_gl.hpp>

#include "cadscene_gl.hpp"

namespace meshlettest {

class ResourcesGL : public Resources
{
public:
  static const int CYCLED_FRAMES = 4;

  struct ProgramIDs
  {
    nvgl::ProgramID draw_object_tris;
    nvgl::ProgramID draw_object_mesh;
    nvgl::ProgramID draw_object_mesh_task;
    nvgl::ProgramID draw_bboxes;
  };

  struct Programs
  {
    GLuint draw_object_tris      = 0;
    GLuint draw_object_mesh      = 0;
    GLuint draw_object_mesh_task = 0;
    GLuint draw_bboxes           = 0;
  };

  struct FrameBuffer
  {
    bool useResolve;
    int  renderWidth;
    int  renderHeight;
    int  supersample;

    GLuint fboScene             = 0;
    GLuint texSceneColor        = 0;
    GLuint texSceneDepthStencil = 0;
  };

  struct Common
  {
    GLuint         standardVao;
    nvgl::Buffer viewBuffer;
    nvgl::Buffer statsBuffer;
    nvgl::Buffer statsReadBuffer;
  };

  struct DrawSetup
  {
    nvgl::Buffer geometryBindings;
  };

  nvgl::ProfilerGL     m_profilerGL;
  nvgl::ProgramManager m_progManager;
  ProgramIDs           m_programids;
  Programs             m_programs;

  Common      m_common;
  DrawSetup   m_setup;
  CadSceneGL  m_scene;
  FrameBuffer m_framebuffer;

  void synchronize() override { glFinish(); }

  bool init(nvgl::ContextWindow* window, nvh::Profiler* profiler);
  void deinit() override;

  bool initPrograms(const std::string& path, const std::string& prepend) override;
  void reloadPrograms(const std::string& prepend) override;
  void updatedPrograms();
  void deinitPrograms();

  bool initFramebuffer(int width, int height, int supersample, bool vsync) override;
  void deinitFramebuffer();

  bool initScene(const CadScene&) override;
  void deinitScene() override;

  void drawBoundingBoxes(const class RenderList* NV_RESTRICT list) const;

  void blitFrame(const FrameConfig& global) override;

  nvmath::mat4f perspectiveProjection(float fovy, float aspect, float nearPlane, float farPlane) const override;

  void getStats(CullStats& stats) override;
  void copyStats() const;

  uvec2 storeU64(GLuint64 address) { return uvec2(address & 0xFFFFFFFF, address >> 32); }

  void enableVertexFormat() const;

  void disableVertexFormat() const;

  static ResourcesGL* get()
  {
    static ResourcesGL resGL;

    return &resGL;
  }
};

}  // namespace meshlettest
