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
    nvgl::ProgramID draw_bboxes;

    nvgl::ProgramID draw_object_mesh;
    nvgl::ProgramID draw_object_mesh_task;
    nvgl::ProgramID draw_object_cull_mesh;
    nvgl::ProgramID draw_object_cull_mesh_task;

  };

  struct Programs
  {
    GLuint draw_object_tris = 0;
    GLuint draw_bboxes      = 0;

    GLuint draw_object_mesh           = 0;
    GLuint draw_object_mesh_task      = 0;
    GLuint draw_object_cull_mesh      = 0;
    GLuint draw_object_cull_mesh_task = 0;
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

  bool init(const nvgl::ContextWindow* window, nvh::Profiler* profiler) override;
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
