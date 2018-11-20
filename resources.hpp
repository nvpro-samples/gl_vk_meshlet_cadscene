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

#include <platform.h>
#include "cadscene.hpp"
#include <nv_helpers_gl/glsltypes_gl.hpp>
#include <nv_helpers/profiler.hpp>
#include <algorithm>

class NVPWindow;
struct ImDrawData;

using namespace nv_math;
#include "common.h"

// allows to use mesh renderers without hw support, simply falls back to regular
#define USE_MESH_FAKE_TEST 0

namespace meshlettest {

  inline size_t alignedSize(size_t sz, size_t align){
    return ((sz + align-1)/align)*align;
  }
  
  struct FrameConfig {
    SceneData             sceneUbo;
    int                   winWidth;
    int                   winHeight;
    const ImDrawData*     imguiDrawData = nullptr;
    bool                  meshletBoxes = false;
  };

  class Resources {
  public:

    static bool     s_vkMeshSupport;
    static bool     s_vkNVglslExtension;
    static uint32_t s_vkDevice;
    static uint32_t s_glDevice;

    bool          m_nativeMeshSupport = false;
    bool          m_fp16 = false;
    bool          m_cullBackFace = false;
    bool          m_clipping = false;

    uint32_t      m_frame = 0;

    uint32_t      m_extraAttributes = 0;
    uint32_t      m_alignedMatrixSize;
    uint32_t      m_alignedMaterialSize;
    uint32_t      m_vertexSize;
    uint32_t      m_vertexAttributeSize;


    virtual void synchronize() {}

    virtual bool init(NVPWindow *window) { return false; }
    virtual void deinit() {}
    
    virtual bool initPrograms(const std::string& path, const std::string& prepend) { return true;}
    virtual void reloadPrograms(const std::string& prepend) {}

    virtual bool initFramebuffer(int width, int height, int supersample, bool vsync) { return true;}

    virtual bool initScene(const CadScene&) { return true; }
    virtual void deinitScene() {}

    virtual void beginFrame() {}
    virtual void blitFrame(const FrameConfig& global) {}
    virtual void endFrame() {}

    virtual void getStats(CullStats& stats) { }

    virtual nv_math::mat4f perspectiveProjection(float fovy, float aspect, float nearPlane, float farPlane) const = 0;

    virtual nv_helpers::Profiler::GPUInterface*  getTimerInterface() { return NULL; }

    inline void initAlignedSizes(unsigned int uboAlignment){
      // FIXME could solve differently

      m_alignedMatrixSize   = (uint32_t)(alignedSize(sizeof(CadScene::MatrixNode), uboAlignment));
      m_alignedMaterialSize = (uint32_t)(alignedSize(sizeof(CadScene::Material), uboAlignment));

      assert(sizeof(CadScene::MatrixNode) == m_alignedMatrixSize);
      assert(sizeof(CadScene::Material) == m_alignedMaterialSize);
    }
  };
}
