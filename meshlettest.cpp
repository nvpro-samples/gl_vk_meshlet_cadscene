/*
 * Copyright (c) 2017-2022, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2017-2022 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */


/* Contact ckubisch@nvidia.com (Christoph Kubisch) for feedback */


#include <imgui/imgui_helper.h>

#if IS_OPENGL
#include <nvgl/appwindowprofiler_gl.hpp>
#include <nvgl/base_gl.hpp>
#include <nvgl/error_gl.hpp>
#include <nvgl/extensions_gl.hpp>
#include <nvgl/glsltypes_gl.hpp>

#define EXE_NAME "gl_meshlet_cadscene"

#elif IS_VULKAN
#include <nvvk/appwindowprofiler_vk.hpp>
#include <nvvk/context_vk.hpp>

#define EXE_NAME "vk_meshlet_cadscene"
#endif


#include <nvh/cameracontrol.hpp>
#include <nvh/fileoperations.hpp>
#include <nvh/geometry.hpp>
#include <nvh/misc.hpp>

#include "nvmeshlet_builder.hpp"

#include "renderer.hpp"


namespace meshlettest {
int const SAMPLE_SIZE_WIDTH(1024);
int const SAMPLE_SIZE_HEIGHT(1024);
int const SAMPLE_MAJOR_VERSION(4);
int const SAMPLE_MINOR_VERSION(5);


// used for loading viewpoint files and material filter files
class SimpleParameterFile
{
public:
  // loads a text file and stores the tokens in a vector per line and
  // a vector of lines
  // everything after a # gets ignored
  SimpleParameterFile(std::string fileName)
  {
    std::ifstream f;
    f.open(fileName);
    if(!f)
      return;

    std::string lineOfFile;
    while(getline(f, lineOfFile))
    {
      if(lineOfFile.length() == 0)
        continue;

      ParameterLine pLine;

      std::stringstream ss(lineOfFile);
      std::string       token;

      while(getline(ss, token, ' '))
      {
        if(token.length() == 0)
          continue;
        if(token[0] == '#')
        {
          // ignore rest of this line
          break;
        }
        Parameter p;
        p.strValue = token;
        pLine.parameter.push_back(p);
      }

      if(pLine.parameter.size() > 0)
      {
        line.push_back(pLine);
      }
    }
    f.close();
  }

  struct Parameter
  {
    // the string of the token
    std::string strValue;

    // returns true on success of storing the token as an int to toFill
    bool toInt(int& toFill)
    {
      bool success = true;
      try
      {
        toFill = std::stoi(strValue);
      }
      catch(...)
      {
        success = false;
      }
      return success;
    }

    // returns true on success of storing the token as a float to toFill
    bool toFloat(float& toFill)
    {
      bool success = true;
      try
      {
        toFill = std::stof(strValue);
      }
      catch(...)
      {
        success = false;
      }
      return success;
    }
  };

  struct ParameterLine
  {
    std::vector<Parameter> parameter;
  };

  std::vector<ParameterLine> line;
};

class Sample
#if IS_OPENGL
    : public nvgl::AppWindowProfilerGL
#elif IS_VULKAN
    : public nvvk::AppWindowProfilerVK
#endif

{

  enum GuiEnums
  {
    GUI_VIEWPOINT,
    GUI_RENDERER,
    GUI_SUPERSAMPLE,
    GUI_MESHLET_VERTICES,
    GUI_MESHLET_PRIMITIVES,
    GUI_TASK_MESHLETS,
  };

public:
  struct Tweak
  {
    bool     usePrimitiveCull  = false;
    bool     useVertexCull     = true;
    bool     useFragBarycentrics = false;
    bool     useBackFaceCull   = true;
    bool     useClipping       = false;
    bool     animate           = false;
    bool     colorize          = false;
    bool     showBboxes        = false;
    bool     showNormals       = false;
    bool     showCulled        = false;
    bool     useStats          = false;
    bool     showPrimIDs       = false;
    float    fov               = 45.0f;
    float    pixelCull         = 0.5f;
    int      renderer          = 0;
    int      viewPoint         = 0;
    int      supersample       = 2;
    int      copies            = 1;
    int      cloneaxisX        = 1;
    int      cloneaxisY        = 1;
    int      cloneaxisZ        = 0;
    uint32_t objectFrom        = 0;
    uint32_t objectNum         = ~0;
    uint32_t maxGroups         = -1;
    int32_t  indexThreshold    = 0;
    uint32_t minTaskMeshlets   = 16;
    uint32_t numTaskMeshlets   = 32;
    vec3f    clipPosition      = vec3f(0.5f);
  };

  struct ViewPoint
  {
    std::string   name;
    nvmath::mat4f mat;
    float         sceneScale;
  };

  bool             m_useUI = true;
  ImGuiH::Registry m_ui;
  double           m_uiTime = 0;

  bool m_supportsFragBarycentrics = false;

  Tweak m_tweak;
  Tweak m_lastTweak;
  bool  m_lastVsync;

  CadScene                  m_scene;
  std::vector<unsigned int> m_renderersSorted;

  Renderer* NV_RESTRICT  m_renderer;
  Resources* NV_RESTRICT m_resources;
  RenderList             m_renderList;
  FrameConfig            m_frameConfig;

  std::string m_shaderprepend;
  std::string m_lastShaderPrepend;
  std::string m_rendererName;

  std::string            m_viewpointFilename = "viewpoints.txt";
  std::vector<ViewPoint> m_viewPoints;

  std::string          m_messageString;
  std::string          m_modelFilename;
  vec3f                m_modelUpVector = vec3f(0, 1, 0);
  CadScene::LoadConfig m_modelConfig;
  CadScene::LoadConfig m_lastModelConfig;

  int    m_frames        = 0;
  double m_lastFrameTime = 0;
  double m_statsCpuTime  = 0;
  double m_statsGpuTime  = 0;

  nvh::CameraControl m_control;

  void setRendererFromName();

  bool initProgram();
  bool initScene(const char* filename, int clones, int cloneaxis);
  bool initFramebuffers(int width, int height);
  void initRenderer(int type);

  void deinitRenderer();

  void saveViewpoint();
  void loadViewpoints();

  void setupConfigParameters();

  std::string getShaderPrepend();

  Sample()
#if IS_OPENGL
      : AppWindowProfilerGL(false)
#elif IS_VULKAN
      : AppWindowProfilerVK(false)
#endif
  {
    m_modelConfig.extraAttributes = 1;
    setupConfigParameters();

#if defined(NDEBUG)
    setVsync(false);
#endif

#if IS_VULKAN
    static VkPhysicalDeviceMeshShaderFeaturesNV meshFeatures = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_NV};
    static VkPhysicalDeviceFloat16Int8FeaturesKHR float16int8Features = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT16_INT8_FEATURES_KHR};
    static VkPhysicalDeviceFragmentShaderBarycentricFeaturesNV baryFeatures = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_NV};
    m_contextInfo.apiMajor              = 1;
    m_contextInfo.apiMinor              = 1;
    m_contextInfo.compatibleDeviceIndex = Resources::s_vkDevice;
    m_contextInfo.addDeviceExtension(VK_NV_MESH_SHADER_EXTENSION_NAME, true, &meshFeatures);
    m_contextInfo.addDeviceExtension(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME, true, &float16int8Features);
    m_contextInfo.addDeviceExtension(VK_NV_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME, true, &baryFeatures);
#endif
  }

public:
  void processUI(int width, int height, double time);

  bool validateConfig() override;

  bool begin() override;
  void think(double time) override;
  void resize(int width, int height) override;

  void postBenchmarkAdvance() override;

  void end() override;

  // return true to prevent m_windowState updates
  bool mouse_pos(int x, int y) override
  {
    if(!m_useUI)
      return false;

    return ImGuiH::mouse_pos(x, y);
  }
  bool mouse_button(int button, int action) override
  {
    if(!m_useUI)
      return false;

    return ImGuiH::mouse_button(button, action);
  }
  bool mouse_wheel(int wheel) override
  {
    if(!m_useUI)
      return false;

    return ImGuiH::mouse_wheel(wheel);
  }
  bool key_char(int key) override
  {
    if(!m_useUI)
      return false;

    return ImGuiH::key_char(key);
  }
  bool key_button(int button, int action, int mods) override
  {
    if(!m_useUI)
      return false;

    return ImGuiH::key_button(button, action, mods);
  }

  template <typename T>
  bool tweakChanged(const T& val) const
  {
    size_t offset = size_t(&val) - size_t(&m_tweak);
    return memcmp(&val, reinterpret_cast<const uint8_t*>(&m_lastTweak) + offset, sizeof(T)) != 0;
  }

  template <typename T>
  bool modelConfigChanged(const T& val) const
  {
    size_t offset = size_t(&val) - size_t(&m_modelConfig);
    return memcmp(&val, reinterpret_cast<const uint8_t*>(&m_lastModelConfig) + offset, sizeof(T)) != 0;
  }
};

std::string Sample::getShaderPrepend()
{
  std::string prepend = m_shaderprepend;
  if(!prepend.empty())
  {
    char* test  = &prepend[0];
    char* found = nullptr;
    while((found = strstr(test, "\\n")) != nullptr)
    {
      found[0] = ' ';
      found[1] = '\n';
      test += 2;
    }
  }

  return prepend + nvh::stringFormat("#define NVMESHLET_VERTEX_COUNT %d\n", m_modelConfig.meshVertexCount)
         + nvh::stringFormat("#define NVMESHLET_PRIMITIVE_COUNT %d\n", m_modelConfig.meshPrimitiveCount)
         + nvh::stringFormat("#define NVMESHLET_ENCODING %d\n", m_modelConfig.meshBuilder == MESHLET_BUILDER_PACKBASIC ? NVMESHLET_ENCODING_PACKBASIC : 0)
         + nvh::stringFormat("#define NVMESHLET_PRIMBITS %d\n",
                             NVMeshlet::findMSB(std::max(32u, m_modelConfig.meshVertexCount) - 1) + 1)
         + nvh::stringFormat("#define NVMESHLET_PER_TASK %d\n", m_tweak.numTaskMeshlets)
         + nvh::stringFormat("#define VERTEX_EXTRAS_COUNT %d\n", m_modelConfig.extraAttributes)
         + nvh::stringFormat("#define USE_VERTEX_CULL %d\n", m_tweak.useVertexCull ? 1 : 0)
         + nvh::stringFormat("#define USE_BARYCENTRIC_SHADING %d\n", m_tweak.useFragBarycentrics && m_supportsFragBarycentrics ? 1 : 0)
         + nvh::stringFormat("#define USE_BACKFACECULL %d\n", m_tweak.useBackFaceCull ? 1 : 0)
         + nvh::stringFormat("#define USE_CLIPPING %d\n", m_tweak.useClipping ? 1 : 0)
         + nvh::stringFormat("#define USE_STATS %d\n", m_tweak.useStats ? 1 : 0)
         + nvh::stringFormat("#define SHOW_PRIMIDS %d\n", m_tweak.showPrimIDs ? 1 : 0)
         + nvh::stringFormat("#define SHOW_BOX %d\n", m_tweak.showBboxes ? 1 : 0)
         + nvh::stringFormat("#define SHOW_NORMAL %d\n", m_tweak.showNormals ? 1 : 0)
         + nvh::stringFormat("#define SHOW_CULLED %d\n", m_tweak.showCulled ? 1 : 0);
}

bool Sample::initProgram()
{

  return true;
}

bool Sample::initScene(const char* filename, int clones, int cloneaxis)
{
  std::string modelFilename(filename);

  if(!nvh::fileExists(filename))
  {
    modelFilename = nvh::getFileName(filename);
    std::vector<std::string> directories;
    directories.push_back(NVPSystem::exePath());
    directories.push_back(NVPSystem::exePath() + "/media");
    directories.push_back(NVPSystem::exePath() + std::string(PROJECT_RELDIRECTORY));
    directories.push_back(NVPSystem::exePath() + std::string(PROJECT_DOWNLOAD_RELDIRECTORY));
    modelFilename = nvh::findFile(modelFilename, directories);
  }

  m_scene.unload();

  bool status = m_scene.loadCSF(modelFilename.c_str(), m_modelConfig, clones, cloneaxis);
  if(status)
  {
    LOGI("\nscene %s\n", filename);
    LOGI("meshlet max vertex:     %2d\n", m_scene.m_cfg.meshVertexCount);
    LOGI("meshlet max primitives: %2d\n", m_scene.m_cfg.meshPrimitiveCount);
    LOGI("extra attributes:       %2d\n", m_scene.m_cfg.extraAttributes);
    LOGI("allow short indices:    %2d\n", m_scene.m_cfg.allowShorts ? 1 : 0);
    LOGI("use fp16 vertices:      %2d\n", m_scene.m_cfg.fp16 ? 1 : 0);
    LOGI("geometries: %9d\n", uint32_t(m_scene.m_geometry.size()));
    LOGI("materials:  %9d\n", uint32_t(m_scene.m_materials.size()));
    LOGI("nodes:      %9d\n", uint32_t(m_scene.m_matrices.size()));
    LOGI("objects:    %9d\n", uint32_t(m_scene.m_objects.size()));
    LOGI("\n");
  }
  else
  {
    LOGW("\ncould not load model %s\n", modelFilename.c_str());
  }

  //if (m_tweak.objectNum == ~0) {
  m_tweak.objectNum = (uint32_t)m_scene.m_objects.size();
  //}
  return status;
}

bool Sample::initFramebuffers(int width, int height)
{
  return m_resources->initFramebuffer(width, height, m_tweak.supersample, getVsync());
}

void Sample::deinitRenderer()
{
  if(m_renderer)
  {
    m_resources->synchronize();
    m_renderer->deinit();
    delete m_renderer;
    m_renderer = nullptr;
  }
}

void Sample::initRenderer(int typesort)
{
  int type = m_renderersSorted[typesort % m_renderersSorted.size()];

  deinitRenderer();

  if(Renderer::getRegistry()[type]->resources() != m_resources)
  {
    if(m_resources)
    {
      m_resources->synchronize();
      m_resources->deinit();
    }
    m_resources                    = Renderer::getRegistry()[type]->resources();
    m_resources->m_cullBackFace    = m_tweak.useBackFaceCull;
    m_resources->m_clipping        = m_tweak.useClipping;
    m_resources->m_extraAttributes = m_modelConfig.extraAttributes;
#if IS_OPENGL
    bool valid = m_resources->init(&m_contextWindow, &m_profiler);
#elif IS_VULKAN
    bool valid = m_resources->init(&m_context, &m_swapChain, &m_profiler);
#endif
    valid = valid
            && m_resources->initFramebuffer(m_windowState.m_swapSize[0], m_windowState.m_swapSize[1],
                                            m_tweak.supersample, getVsync());
    valid = valid && m_resources->initPrograms(exePath(), getShaderPrepend());
    valid = valid && m_resources->initScene(m_scene);

    if(!valid)
    {
      LOGE("failed to initialize resources\n");
      exit(-1);
    }

    m_resources->m_frame = 0;
  }

  {
    RenderList::Config config;
    config.objectFrom      = m_tweak.objectFrom;
    config.objectNum       = m_tweak.objectNum;
    config.taskMinMeshlets = m_tweak.minTaskMeshlets;
    config.taskNumMeshlets = m_tweak.numTaskMeshlets;
    config.indexThreshold  = m_tweak.indexThreshold;
    config.strategy        = RenderList::STRATEGY_SINGLE;

    m_renderList.setup(&m_scene, config);
  }
  {
    Renderer::Config config;
    config.useCulling = m_tweak.usePrimitiveCull;

    LOGI("renderer: %s\n", Renderer::getRegistry()[type]->name());
    m_renderer = Renderer::getRegistry()[type]->create();
    m_renderer->init(&m_renderList, m_resources, config);
  }
}

void Sample::loadViewpoints()
{
  SimpleParameterFile vpParameters(m_viewpointFilename);
  for(SimpleParameterFile::ParameterLine line : vpParameters.line)
  {
    // name + 16 for the matrix + optional scale
    if(line.parameter.size() == 17 || line.parameter.size() == 18)
    {
      bool      lineIsOK = true;
      ViewPoint vp;
      vp.name = line.parameter[0].strValue;

      // read matrix
      for(auto i = 0; i < 16; ++i)
      {
        bool valueIsFloat = line.parameter[1 + i].toFloat(vp.mat.mat_array[i]);

        lineIsOK = lineIsOK & valueIsFloat;
      }

      // optionally realy scale
      if(line.parameter.size() == 18)
      {
        lineIsOK = lineIsOK & line.parameter[17].toFloat(vp.sceneScale);
      }
      else
      {
        vp.sceneScale = 1.0f;
      }

      // only save if all parameters were read correctly
      if(lineIsOK)
      {
        m_viewPoints.push_back(vp);
      }
    }
  }
}


bool Sample::begin()
{
#if IS_OPENGL
  m_supportsFragBarycentrics = has_GL_NV_fragment_shader_barycentric != 0;
#elif IS_VULKAN
  m_supportsFragBarycentrics = m_context.hasDeviceExtension(VK_NV_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME);
#endif

  m_profilerPrint = false;
  m_timeInTitle   = true;

  m_renderer  = nullptr;
  m_resources = nullptr;

  bool validated(true);
  validated = validated && initProgram();
  validated = validated
              && initScene(m_modelFilename.c_str(), m_tweak.copies - 1,
                           (m_tweak.cloneaxisX << 0) | (m_tweak.cloneaxisY << 1) | (m_tweak.cloneaxisZ << 2));

  const Renderer::Registry registry = Renderer::getRegistry();
  for(size_t i = 0; i < registry.size(); i++)
  {
#if IS_OPENGL
    if(registry[i]->isAvailable(&m_contextWindow))
#elif IS_VULKAN
    if(registry[i]->isAvailable(&m_context))
#endif
    {
      uint sortkey = uint(i);
      sortkey |= registry[i]->priority() << 16;
      m_renderersSorted.push_back(sortkey);
    }
  }

  if(m_renderersSorted.empty())
  {
    LOGE("No renderers available\n");
    return false;
  }

  std::sort(m_renderersSorted.begin(), m_renderersSorted.end());

  for(size_t i = 0; i < m_renderersSorted.size(); i++)
  {
    m_renderersSorted[i] &= 0xFFFF;
    LOGI("renderer %d: %s\n", uint32_t(i), registry[m_renderersSorted[i]]->name());
  }

  setRendererFromName();

  loadViewpoints();

  ImGuiH::Init(m_windowState.m_winSize[0], m_windowState.m_winSize[1], this);
  if(m_useUI)
  {
    auto& imgui_io = ImGui::GetIO();

    for(size_t i = 0; i < m_renderersSorted.size(); i++)
    {
      m_ui.enumAdd(GUI_RENDERER, int(i), registry[m_renderersSorted[i]]->name());
    }
    for(auto it = m_viewPoints.begin(); it != m_viewPoints.end(); it++)
    {
      m_ui.enumAdd(GUI_VIEWPOINT, int(it - m_viewPoints.begin()), it->name.c_str());
    }
    if(m_viewPoints.empty())
    {
      m_ui.enumAdd(GUI_VIEWPOINT, 0, "default");
    }

    m_ui.enumAdd(GUI_SUPERSAMPLE, 1, "none");
    m_ui.enumAdd(GUI_SUPERSAMPLE, 2, "4x");
    m_ui.enumAdd(GUI_SUPERSAMPLE, 3, "9x");
    m_ui.enumAdd(GUI_SUPERSAMPLE, 4, "16x");

    // must be multiple of 32 (subgroup size)
    m_ui.enumAdd(GUI_MESHLET_VERTICES, 32, "32");
    m_ui.enumAdd(GUI_MESHLET_VERTICES, 64, "64");
    m_ui.enumAdd(GUI_MESHLET_VERTICES, 96, "96");
    m_ui.enumAdd(GUI_MESHLET_VERTICES, 128, "128");

    m_ui.enumAdd(GUI_TASK_MESHLETS, 32, "32");
    m_ui.enumAdd(GUI_TASK_MESHLETS, 64, "64");
    m_ui.enumAdd(GUI_TASK_MESHLETS, 96, "96");
    m_ui.enumAdd(GUI_TASK_MESHLETS, 128, "128");

    // the 40,84,126 are tuned for the allocation granularity
    m_ui.enumAdd(GUI_MESHLET_PRIMITIVES, 32, "32");
    m_ui.enumAdd(GUI_MESHLET_PRIMITIVES, 40, "40");
    m_ui.enumAdd(GUI_MESHLET_PRIMITIVES, 64, "64");
    m_ui.enumAdd(GUI_MESHLET_PRIMITIVES, 84, "84");
    m_ui.enumAdd(GUI_MESHLET_PRIMITIVES, 96, "96");
    m_ui.enumAdd(GUI_MESHLET_PRIMITIVES, 126, "126");
  }

  m_control.m_sceneUp        = m_modelUpVector;
  m_control.m_sceneOrbit     = nvmath::vec3f(m_scene.m_bbox.max + m_scene.m_bbox.min) * 0.5f;
  m_control.m_sceneDimension = nvmath::length((m_scene.m_bbox.max - m_scene.m_bbox.min));
  m_control.m_viewMatrix = nvmath::look_at(m_control.m_sceneOrbit - (-vec3(1, 1, 1) * m_control.m_sceneDimension * 0.5f),
                                           m_control.m_sceneOrbit, m_modelUpVector);

  m_frameConfig.sceneUbo.wLightPos   = (m_scene.m_bbox.max + m_scene.m_bbox.min) * 0.5f + m_control.m_sceneDimension;
  m_frameConfig.sceneUbo.wLightPos.w = 1.0;
  m_frameConfig.sceneUbo.colorize    = 0;

  initRenderer(m_tweak.renderer);

  if(!m_viewPoints.empty())
  {
    m_control.m_viewMatrix = m_viewPoints[m_tweak.viewPoint].mat;
  }
  else
  {
    m_tweak.viewPoint = 0;
  }

  m_lastTweak         = m_tweak;
  m_lastModelConfig   = m_modelConfig;
  m_lastShaderPrepend = m_shaderprepend;

  return validated;
}


void Sample::end()
{
  deinitRenderer();
  if(m_resources)
  {
    m_resources->deinit();
  }
}

void Sample::processUI(int width, int height, double time)
{
  // Update imgui configuration
  auto& imgui_io       = ImGui::GetIO();
  imgui_io.DeltaTime   = static_cast<float>(time - m_uiTime);
  imgui_io.DisplaySize = ImVec2(width, height);

  m_uiTime = time;

  ImGui::NewFrame();
  ImGui::SetNextWindowPos(ImVec2(5, 5));
  ImGui::SetNextWindowSize(ImGuiH::dpiScaled(290, 800), ImGuiCond_FirstUseEver);

  if(ImGui::Begin("NVIDIA " EXE_NAME, nullptr))
  {
    ImGui::PushItemWidth(ImGuiH::dpiScaled(120));
    ImGui::Separator();
    if(!m_messageString.empty())
    {
      ImGui::Text("%s", m_messageString.c_str());
      ImGui::Separator();
    }

    m_ui.enumCombobox(GUI_RENDERER, "renderer", &m_tweak.renderer);
    m_ui.enumCombobox(GUI_VIEWPOINT, "viewpoint", &m_tweak.viewPoint);
    ImGui::SliderFloat("fov", &m_tweak.fov, 1, 120, "%.0f");
    if(ImGui::CollapsingHeader("Mesh Shading Settings", ImGuiTreeNodeFlags_DefaultOpen))
    {
      m_ui.enumCombobox(GUI_MESHLET_VERTICES, "meshlet vertices", &m_modelConfig.meshVertexCount);
      m_ui.enumCombobox(GUI_MESHLET_PRIMITIVES, "meshlet primitives", &m_modelConfig.meshPrimitiveCount);
      ImGui::Checkbox("(mesh) colorize by meshlet", &m_tweak.colorize);

      m_ui.enumCombobox(GUI_MESHLET_VERTICES, "task meshlet count", &m_tweak.numTaskMeshlets);
      ImGuiH::InputIntClamped("task min. meshlets\n0 disables task stage", &m_tweak.minTaskMeshlets, 0, 256, 1, 16, ImGuiInputTextFlags_EnterReturnsTrue);
      ImGui::SliderFloat("(task) pixel cull", &m_tweak.pixelCull, 0.0f, 1.0f, "%.2f");
      ImGui::Checkbox("(mesh) use per-primitive culling ", &m_tweak.usePrimitiveCull);
      ImGui::Checkbox("- also use per-vertex culling", &m_tweak.useVertexCull);
      if(m_supportsFragBarycentrics)
      {
        ImGui::Checkbox("(mesh) use fragment barycentrics ", &m_tweak.useFragBarycentrics);
      }
    }
    if(ImGui::CollapsingHeader("Render Settings"))
    {
      ImGui::Checkbox("show primitive ids", &m_tweak.showPrimIDs);
      ImGui::Checkbox("show meshlet bboxes", &m_tweak.showBboxes);
      ImGui::Checkbox("show meshlet normals", &m_tweak.showNormals);
      ImGui::Checkbox("- culled bboxes/normals", &m_tweak.showCulled);
      ImGui::NewLine();
      ImGui::Checkbox("use backface culling ", &m_tweak.useBackFaceCull);
      ImGui::Checkbox("use clipping planes", &m_tweak.useClipping);
      ImGui::SliderFloat3("clip position", m_tweak.clipPosition.vec_array, 0.01f, 1.01, "%.2f");

#if 0
      ImGui::Separator();
      ImGuiH::InputIntClamped("objectFrom", &m_tweak.objectFrom, 0, (int)m_scene.m_objects.size()-1);
      ImGuiH::InputIntClamped("objectNum", &m_tweak.objectNum, 0, (int)m_scene.m_objects.size());
      ImGui::InputInt("indexThreshold", &m_tweak.indexThreshold);
#endif
    }

    {
      int avg = 50;

      if(m_lastFrameTime == 0)
      {
        m_lastFrameTime = time;
        m_frames        = -1;
      }

      if(m_frames > 4)
      {
        double curavg = (time - m_lastFrameTime) / m_frames;
        if(curavg > 1.0 / 30.0)
        {
          avg = 10;
        }
      }

      if(m_profiler.getTotalFrames() % avg == avg - 1)
      {
        nvh::Profiler::TimerInfo info;
        m_profiler.getTimerInfo("Render", info);
        m_statsCpuTime  = info.cpu.average;
        m_statsGpuTime  = info.gpu.average;
        m_lastFrameTime = time;
        m_frames        = -1;
      }

      m_frames++;

      float gpuTimeF = float(m_statsGpuTime);
      if(ImGui::CollapsingHeader("Basic Stats", ImGuiTreeNodeFlags_DefaultOpen))
      {
        ImGui::Text("         Render GPU [ms]: %2.3f", gpuTimeF / 1000.0f);
        ImGui::Text("Original Index Size [MB]: %4zu", m_scene.m_iboSize / (1024 * 1024));
        ImGui::Text("       Meshlet Size [MB]: %4zu", m_scene.m_meshSize / (1024 * 1024));
      }
    }

    m_tweak.useStats = ImGui::CollapsingHeader("Detailed Stats (costs perf)");
    if(m_tweak.useStats)
    {
      CullStats stats;
      m_resources->getStats(stats);
      ImGui::Text("task total:  %9d", stats.tasksInput);
      //ImGui::Text("task output: %9d - %2.1f", stats.tasksOutput, double(stats.tasksOutput)/double(stats.tasksInput)*100);
      ImGui::Text("mesh total:  %9d", stats.meshletsInput);
      ImGui::Text("mesh output: %9d - %2.1f", stats.meshletsOutput,
                  double(stats.meshletsOutput) / double(stats.meshletsInput) * 100);
      ImGui::Text("tri  total:  %9d", stats.trisInput);
      ImGui::Text("tri  output: %9d - %2.1f", stats.trisOutput, double(stats.trisOutput) / double(stats.trisInput) * 100);
      ImGui::Text("vert input:  %9d", stats.attrInput);
      ImGui::Text("attr read:   %9d - %2.1f", stats.attrOutput, double(stats.attrOutput) / double(stats.attrInput) * 100);
    }

    if(ImGui::CollapsingHeader("Model Settings"))
    {
      ImGui::Checkbox("use fp16 vtx attribs", &m_modelConfig.fp16);
      ImGuiH::InputIntClamped("extra vec4 attribs", &m_modelConfig.extraAttributes, 0, 7);
      ImGuiH::InputIntClamped("model copies", &m_tweak.copies, 1, 256, 1, 10, ImGuiInputTextFlags_EnterReturnsTrue);
    }

    if(ImGui::CollapsingHeader("Misc"))
    {
      m_ui.enumCombobox(GUI_SUPERSAMPLE, "super resolution", &m_tweak.supersample);
    }
  }
  ImGui::End();
}


void Sample::think(double time)
{
  int width  = m_windowState.m_swapSize[0];
  int height = m_windowState.m_swapSize[1];

  if(m_useUI)
  {
    processUI(width, height, time);
  }

  m_control.processActions(m_windowState.m_winSize,
                           nvmath::vec2f(m_windowState.m_mouseCurrent[0], m_windowState.m_mouseCurrent[1]),
                           m_windowState.m_mouseButtonFlags, m_windowState.m_mouseWheel);


  // trigger recompile of shaders
  if(m_windowState.onPress(KEY_R) || tweakChanged(m_tweak.useBackFaceCull) || tweakChanged(m_tweak.useClipping)
     || tweakChanged(m_tweak.useStats) || tweakChanged(m_tweak.showBboxes) || tweakChanged(m_tweak.showNormals)
     || tweakChanged(m_tweak.showCulled) || tweakChanged(m_tweak.showPrimIDs) || tweakChanged(m_tweak.numTaskMeshlets)
     || tweakChanged(m_tweak.useFragBarycentrics) || tweakChanged(m_tweak.useVertexCull)
     || modelConfigChanged(m_modelConfig.extraAttributes) || modelConfigChanged(m_modelConfig.meshPrimitiveCount)
     || modelConfigChanged(m_modelConfig.meshVertexCount) || m_shaderprepend != m_lastShaderPrepend)
  {
    m_resources->synchronize();
    m_resources->m_cullBackFace = m_tweak.useBackFaceCull;
    m_resources->m_clipping     = m_tweak.useClipping;
    m_resources->reloadPrograms(getShaderPrepend());
  }
  else if(m_windowState.onPress(KEY_C))
  {
    saveViewpoint();
  }

  if(tweakChanged(m_tweak.supersample) || getVsync() != m_lastVsync)
  {
    m_lastVsync = getVsync();
    m_resources->initFramebuffer(width, height, m_tweak.supersample, getVsync());
  }

  bool sceneChanged = false;
  if(tweakChanged(m_tweak.copies) || tweakChanged(m_tweak.cloneaxisX) || tweakChanged(m_tweak.cloneaxisY)
     || tweakChanged(m_tweak.cloneaxisZ) || memcmp(&m_modelConfig, &m_lastModelConfig, sizeof(m_modelConfig)))
  {
    sceneChanged = true;
    m_resources->synchronize();
    deinitRenderer();
    m_resources->deinitScene();
    initScene(m_modelFilename.c_str(), m_tweak.copies - 1,
              (m_tweak.cloneaxisX << 0) | (m_tweak.cloneaxisY << 1) | (m_tweak.cloneaxisZ << 2));
    m_resources->initScene(m_scene);
  }

  if(sceneChanged || tweakChanged(m_tweak.renderer) || tweakChanged(m_tweak.objectFrom)
     || tweakChanged(m_tweak.objectNum) || tweakChanged(m_tweak.useClipping) || tweakChanged(m_tweak.useStats)
     || tweakChanged(m_tweak.maxGroups) || tweakChanged(m_tweak.indexThreshold) || tweakChanged(m_tweak.minTaskMeshlets)
     || tweakChanged(m_tweak.usePrimitiveCull) || tweakChanged(m_tweak.numTaskMeshlets))
  {
    m_resources->synchronize();
    initRenderer(m_tweak.renderer);
  }

  if(tweakChanged(m_tweak.viewPoint))
  {
    m_control.m_viewMatrix = m_viewPoints[m_tweak.viewPoint].mat;
  }

  m_resources->beginFrame();

  {
    m_frameConfig.winWidth     = width;
    m_frameConfig.winHeight    = height;
    m_frameConfig.meshletBoxes = m_tweak.showBboxes || m_tweak.showNormals;

    SceneData& sceneUbo = m_frameConfig.sceneUbo;

    sceneUbo.viewport         = ivec2(width * m_tweak.supersample, height * m_tweak.supersample);
    sceneUbo.viewportf        = vec2(width * m_tweak.supersample, height * m_tweak.supersample);
    sceneUbo.viewportTaskCull = sceneUbo.viewportf * m_tweak.pixelCull;
    sceneUbo.colorize         = m_tweak.colorize ? 1 : 0;

    if(m_tweak.animate)
    {
      float         t        = float(time);
      nvmath::quatf quat     = nvmath::axis_to_quat(m_modelUpVector, t * 0.5f);
      mat4          rotator  = nvmath::quat_2_mat(quat);
      vec3          dir      = rotator * (-vec3(1, 1, 1));
      float         distance = 0.4f + sinf(t) * 0.2f;
      m_control.m_viewMatrix = nvmath::look_at(m_control.m_sceneOrbit - (dir * m_control.m_sceneDimension * distance),
                                               m_control.m_sceneOrbit, m_modelUpVector);
    }

    nvmath::mat4 projection =
        m_resources->perspectiveProjection(m_tweak.fov, float(width) / float(height),
                                           m_control.m_sceneDimension * 0.001f, m_control.m_sceneDimension * 10.0f);
    nvmath::mat4 view  = m_control.m_viewMatrix;
    nvmath::mat4 viewI = nvmath::invert(view);

    sceneUbo.viewProjMatrix = projection * view;
    sceneUbo.viewMatrix     = view;
    sceneUbo.viewMatrixIT   = nvmath::transpose(viewI);

    sceneUbo.viewPos = sceneUbo.viewMatrixIT.row(3);
    sceneUbo.viewDir = -view.row(2);

    sceneUbo.wLightPos   = sceneUbo.viewMatrixIT.row(3);
    sceneUbo.wLightPos.w = 1.0;

    nvmath::vec3 viewDir   = view.row(2);
    nvmath::vec3 sideDir   = -view.row(0);
    nvmath::vec3 sideUpDir = view.row(1);
    sceneUbo.wLightPos += (sideDir + sideUpDir + viewDir * 0.25f) * m_control.m_sceneDimension * 0.25f;

    sceneUbo.wClipPlanes[0] =
        vec4f(-1, 0, 0, nvmath::lerp(m_tweak.clipPosition.x, m_scene.m_bboxInstanced.min.x, m_scene.m_bboxInstanced.max.x));
    sceneUbo.wClipPlanes[1] =
        vec4f(0, -1, 0, nvmath::lerp(m_tweak.clipPosition.y, m_scene.m_bboxInstanced.min.y, m_scene.m_bboxInstanced.max.y));
    sceneUbo.wClipPlanes[2] =
        vec4f(0, 0, -1, nvmath::lerp(m_tweak.clipPosition.z, m_scene.m_bboxInstanced.min.z, m_scene.m_bboxInstanced.max.z));
  }


  {
    m_renderer->draw(m_frameConfig);
  }

  {
    if(m_useUI)
    {
      ImGui::Render();
      m_frameConfig.imguiDrawData = ImGui::GetDrawData();
    }
    else
    {
      m_frameConfig.imguiDrawData = nullptr;
    }

    m_resources->blitFrame(m_frameConfig);
  }

  m_resources->endFrame();
  m_resources->m_frame++;

  if(m_useUI)
  {
    ImGui::EndFrame();
  }

  m_lastTweak       = m_tweak;
  m_lastModelConfig = m_modelConfig;
  m_shaderprepend   = m_lastShaderPrepend;
}

void Sample::resize(int width, int height)
{
  initFramebuffers(width, height);
}

void Sample::postBenchmarkAdvance()
{
  setRendererFromName();
}

void Sample::saveViewpoint()
{
  int idx = int(m_viewPoints.size());

  ViewPoint vp;
  vp.mat        = m_control.m_viewMatrix;
  vp.sceneScale = 1.0;
  vp.name       = nvh::stringFormat("ViewPoint%d", idx);
  m_ui.enumAdd(GUI_VIEWPOINT, idx, vp.name.c_str());

  m_viewPoints.push_back(vp);
  m_tweak.viewPoint = idx;

  std::ofstream f;
  f.open(m_viewpointFilename, std::ios_base::app | std::ios_base::out);
  if(f)
  {
    f << vp.name << " ";
    for(auto i = 0; i < 16; ++i)
    {
      f << vp.mat.mat_array[i] << " ";
    }
    f << vp.sceneScale;
    f << "\n";
    f.close();
  }

  LOGI("viewpoint file updated: %s\n", m_viewpointFilename.c_str());
}

static std::string addPath(std::string const& defaultPath, std::string const& filename)
{
  if(
#ifdef _WIN32
      filename.find(':') != std::string::npos
#else
      !filename.empty() && filename[0] == '/'
#endif
  )
  {
    return filename;
  }
  else
  {
    return defaultPath + "/" + filename;
  }
}

static bool endsWith(std::string const& s, std::string const& end)
{
  if(s.length() >= end.length())
  {
    return (0 == s.compare(s.length() - end.length(), end.length(), end));
  }
  else
  {
    return false;
  }
}

void Sample::setupConfigParameters()
{
  m_parameterList.addFilename(".csf", &m_modelFilename);
  m_parameterList.addFilename(".csf.gz", &m_modelFilename);
  m_parameterList.addFilename(".gltf", &m_modelFilename);

  m_parameterList.add("vkdevice", &Resources::s_vkDevice);
  m_parameterList.add("gldevice", &Resources::s_glDevice);

  m_parameterList.addFilename("viewpoints", &m_viewpointFilename);
  m_parameterList.add("viewpoint", &m_tweak.viewPoint);
  m_parameterList.add("animate", &m_tweak.animate);

  m_parameterList.add("noui", &m_useUI, false);

  m_parameterList.add("copies", &m_tweak.copies);

  m_parameterList.add("renderer", (uint32_t*)&m_tweak.renderer);
  m_parameterList.add("renderernamed", &m_rendererName);
  m_parameterList.add("supersample", &m_tweak.supersample);
  m_parameterList.add("superresolution", &m_tweak.supersample);

  m_parameterList.add("fov", &m_tweak.fov);
  m_parameterList.add("scale", &m_modelConfig.scale);
  m_parameterList.add("upvector", &m_modelUpVector.x, nullptr, 3);
  m_parameterList.add("verbose", &m_modelConfig.verbose);
  m_parameterList.add("allowshorts", &m_modelConfig.allowShorts);
  m_parameterList.add("fp16vertices", &m_modelConfig.fp16);
  m_parameterList.add("extraattributes", &m_modelConfig.extraAttributes);
  m_parameterList.add("colorizeextra", &m_modelConfig.colorizeExtra);

  m_parameterList.add("objectfirst", &m_tweak.objectFrom);
  m_parameterList.add("objectnum", &m_tweak.objectNum);
  m_parameterList.add("maxgroups", &m_tweak.maxGroups);
  m_parameterList.add("indexthreshold", &m_tweak.indexThreshold);
  m_parameterList.add("taskminmeshlets", &m_tweak.minTaskMeshlets);
  m_parameterList.add("taskpixelcull", &m_tweak.pixelCull);

  m_parameterList.add("shaderprepend", &m_shaderprepend);

  m_parameterList.add("meshlet", &m_modelConfig.meshVertexCount, nullptr, 2);
  m_parameterList.add("primitivecull", &m_tweak.usePrimitiveCull);
  m_parameterList.add("vertexcull", &m_tweak.useVertexCull);
  m_parameterList.add("backfacecull", &m_tweak.useBackFaceCull);

  m_parameterList.add("showbbox", &m_tweak.showBboxes);
  m_parameterList.add("shownormals", &m_tweak.showNormals);
  m_parameterList.add("showculled", &m_tweak.showCulled);

  m_parameterList.add("fragbarycentrics", &m_tweak.useFragBarycentrics);

  m_parameterList.add("primids", &m_tweak.showPrimIDs);

  m_parameterList.add("stats", &m_tweak.useStats);

  m_parameterList.add("clipping", &m_tweak.useClipping);
  m_parameterList.add("clippos", m_tweak.clipPosition.vec_array, nullptr, 3);

  m_parameterList.add("message", &m_messageString);
}


bool Sample::validateConfig()
{
  if(m_modelFilename.empty())
  {
    std::vector<std::string> directories;
    directories.push_back(NVPSystem::exePath());
    directories.push_back(NVPSystem::exePath() + "/media");
    directories.push_back(NVPSystem::exePath() + std::string(PROJECT_RELDIRECTORY));
    directories.push_back(NVPSystem::exePath() + std::string(PROJECT_DOWNLOAD_RELDIRECTORY));
    std::string configFile = nvh::findFile("worldcar_meshlet.cfg", directories);
    if(!configFile.empty())
    {
      parseConfigFile(configFile.c_str());
    }
  }

  if(m_modelFilename.empty())
  {
    LOGI("no modelfile specified\n");
    LOGI("exe <filename.csf/cfg> parameters...\n");
    m_parameterList.print();
    return false;
  }
  return true;
}

void Sample::setRendererFromName()
{
  if(!m_rendererName.empty())
  {
    const Renderer::Registry registry = Renderer::getRegistry();
    for(size_t i = 0; i < m_renderersSorted.size(); i++)
    {
      if(strcmp(m_rendererName.c_str(), registry[m_renderersSorted[i]]->name()) == 0)
      {
        m_tweak.renderer = int(i);
      }
    }
  }
}

}  // namespace meshlettest

using namespace meshlettest;

#include <omp.h>
#include <thread>

int main(int argc, const char** argv)
{
  NVPSystem system(EXE_NAME);

  omp_set_num_threads(std::thread::hardware_concurrency());

  Sample sample;

  return sample.run(EXE_NAME, argc, argv, SAMPLE_SIZE_WIDTH, SAMPLE_SIZE_HEIGHT);
}
