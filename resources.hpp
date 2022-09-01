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

#include "cadscene.hpp"
#include <nvgl/glsltypes_gl.hpp>
#include <nvh/profiler.hpp>
#include <platform.h>
#if IS_OPENGL
#include <nvgl/contextwindow_gl.hpp>
#elif IS_VULKAN
#include <nvvk/context_vk.hpp>
#include <nvvk/swapchain_vk.hpp>
#endif


#include <algorithm>

struct ImDrawData;

#include "common.h"

namespace meshlettest {

inline size_t alignedSize(size_t sz, size_t align)
{
  return ((sz + align - 1) / align) * align;
}

struct FrameConfig
{
  SceneData   sceneUbo;
  int         winWidth{};
  int         winHeight{};
  ImDrawData* imguiDrawData = nullptr;
  bool        meshletBoxes  = false;
};

class Resources
{
public:
  static uint32_t s_vkDevice;
  static uint32_t s_glDevice;

  bool m_nativeMeshSupport = false;
  bool m_fp16              = false;
  bool m_cullBackFace      = false;
  bool m_clipping          = false;

  uint32_t m_frame = 0;

  uint32_t m_extraAttributes = 0;
  uint32_t m_alignedMatrixSize{};
  uint32_t m_alignedMaterialSize{};
  uint32_t m_vertexSize{};
  uint32_t m_vertexAttributeSize{};


  virtual void synchronize() {}

  // Can't virtualize it anymore :-(
#if IS_OPENGL
  virtual bool init(const nvgl::ContextWindow* window, nvh::Profiler* profiler) { return false; }
#elif IS_VULKAN
  virtual bool init(const nvvk::Context* context, const nvvk::SwapChain* swapChain, nvh::Profiler* profiler)
  {
    return false;
  }
#endif
  virtual void deinit() {}

  virtual bool initPrograms(const std::string& path, const std::string& prepend) { return true; }
  virtual void reloadPrograms(const std::string& prepend) {}

  virtual bool initFramebuffer(int width, int height, int supersample, bool vsync) { return true; }

  virtual bool initScene(const CadScene&) { return true; }
  virtual void deinitScene() {}

  virtual void beginFrame() {}
  virtual void blitFrame(const FrameConfig& global) {}
  virtual void endFrame() {}

  virtual void getStats(CullStats& stats) {}

  [[nodiscard]] virtual nvmath::mat4f perspectiveProjection(float fovy, float aspect, float nearPlane, float farPlane) const = 0;

  inline void initAlignedSizes(unsigned int uboAlignment)
  {
    // FIXME could solve differently

    m_alignedMatrixSize   = (uint32_t)(alignedSize(sizeof(CadScene::MatrixNode), uboAlignment));
    m_alignedMaterialSize = (uint32_t)(alignedSize(sizeof(CadScene::Material), uboAlignment));

    assert(sizeof(CadScene::MatrixNode) == m_alignedMatrixSize);
    assert(sizeof(CadScene::Material) == m_alignedMaterialSize);
  }
};
}  // namespace meshlettest
