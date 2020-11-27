/* Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

/* Contact ckubisch@nvidia.com (Christoph Kubisch) for feedback */


#include <imgui/imgui_helper.h>

#if HAS_OPENGL
#include <nvgl/appwindowprofiler_gl.hpp>
#include <nvgl/base_gl.hpp>
#include <nvgl/error_gl.hpp>
#include <nvgl/extensions_gl.hpp>
#include <nvgl/glsltypes_gl.hpp>
#else
#include <nvvk/appwindowprofiler_vk.hpp>
#endif

#include <nvvk/context_vk.hpp>

#include <nvh/cameracontrol.hpp>
#include <nvh/geometry.hpp>
#include <nvh/fileoperations.hpp>
#include <nvh/misc.hpp>

#include "nvmeshlet_builder.hpp"

#include "renderer.hpp"

extern bool vulkanIsExtensionSupported(uint32_t, const char* name);

namespace meshlettest {
int const SAMPLE_SIZE_WIDTH(1024);
int const SAMPLE_SIZE_HEIGHT(1024);
int const SAMPLE_MAJOR_VERSION(4);
int const SAMPLE_MINOR_VERSION(5);

void setupVulkanContextInfo(nvvk::ContextCreateInfo& info)
{
  static VkPhysicalDeviceMeshShaderFeaturesNV meshFeatures = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_NV};
  static VkPhysicalDeviceFloat16Int8FeaturesKHR float16int8Features = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT16_INT8_FEATURES_KHR};
  info.apiMajor                                                     = 1;
  info.apiMinor                                                     = 1;
  info.compatibleDeviceIndex                                                       = Resources::s_vkDevice;
#if HAS_OPENGL
  // not compatible with GL extension mechanism
  info.removeInstanceLayer("VK_LAYER_KHRONOS_validation");
#endif
  info.addDeviceExtension(VK_NV_GLSL_SHADER_EXTENSION_NAME, true);  // flag optional, driver still supports it
  info.addDeviceExtension(VK_NV_MESH_SHADER_EXTENSION_NAME, true, &meshFeatures);
  info.addDeviceExtension(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME, true, &float16int8Features);
}

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
#if HAS_OPENGL
    : public nvgl::AppWindowProfilerGL
#else
    : public nvvk::AppWindowProfilerVK
#endif

{

  enum GuiEnums
  {
    GUI_VIEWPOINT,
    GUI_RENDERER,
    GUI_SUPERSAMPLE,
  };

public:
  struct Tweak
  {
    bool     useMeshShaderCull = false;
    bool     useBackFaceCull   = true;
    bool     useClipping       = false;
    bool     animate           = false;
    bool     colorize          = false;
    bool     showBboxes        = false;
    bool     showNormals       = false;
    bool     showCulled        = false;
    bool     useStats          = false;
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

  Tweak m_tweak;
  Tweak m_lastTweak;
  bool  m_lastVsync;

  CadScene                  m_scene;
  std::vector<unsigned int> m_renderersSorted;

  Renderer* NV_RESTRICT m_renderer;
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
#if HAS_OPENGL
      : AppWindowProfilerGL(false, true)
#else
      : AppWindowProfilerVK(false, true)
#endif
  {
    m_modelConfig.extraAttributes = 1;
    setupConfigParameters();

#if defined(NDEBUG)
    setVsync(false);
#endif

#if !HAS_OPENGL
    setupVulkanContextInfo(m_contextInfo);
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
         + nvh::stringFormat("#define NVMESHLET_USE_PACKBASIC %d\n", m_modelConfig.meshBuilder == MESHLET_BUILDER_PACKBASIC ? 1 : 0)
         + nvh::stringFormat("#define NVMESHLET_USE_ARRAYS %d\n", m_modelConfig.meshBuilder == MESHLET_BUILDER_ARRAYS ? 1 : 0)
         + nvh::stringFormat("#define NVMESHLET_PRIMBITS %d\n", NVMeshlet::findMSB(std::max(32u, m_modelConfig.meshVertexCount) - 1) + 1)
         + nvh::stringFormat("#define EXTRA_ATTRIBUTES %d\n", m_modelConfig.extraAttributes)
         + nvh::stringFormat("#define USE_MESH_SHADERCULL %d\n", m_tweak.useMeshShaderCull ? 1 : 0)
         + nvh::stringFormat("#define USE_BACKFACECULL %d\n", m_tweak.useBackFaceCull ? 1 : 0)
         + nvh::stringFormat("#define USE_CLIPPING %d\n", m_tweak.useClipping ? 1 : 0)
         + nvh::stringFormat("#define USE_STATS %d\n", m_tweak.useStats ? 1 : 0)
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
    std::vector<std::string> searchPaths;
    searchPaths.push_back("./");
    searchPaths.push_back(exePath() + PROJECT_RELDIRECTORY + "/");
    searchPaths.push_back(exePath() + PROJECT_DOWNLOAD_RELDIRECTORY + "/");
    modelFilename = nvh::findFile(modelFilename, searchPaths);
  }

  m_scene.unload();

  bool status = m_scene.loadCSF(modelFilename.c_str(), m_modelConfig, clones, cloneaxis);
  if(!status)
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
    m_renderer = NULL;
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
#if HAS_OPENGL
    bool valid = m_resources->init(&m_contextWindow, &m_profiler);
#else
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
    config.minTaskMeshlets = m_tweak.minTaskMeshlets;
    config.indexThreshold  = m_tweak.indexThreshold;
    config.strategy        = RenderList::STRATEGY_SINGLE;

    m_renderList.setup(&m_scene, config);
  }
  {
    Renderer::Config config;

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
  Resources::s_vkMeshSupport = vulkanIsExtensionSupported(Resources::s_vkDevice, VK_NV_MESH_SHADER_EXTENSION_NAME);

  m_profilerPrint = false;
  m_timeInTitle   = true;

  m_renderer  = NULL;
  m_resources = NULL;

  bool validated(true);
  validated = validated && initProgram();
  validated = validated
              && initScene(m_modelFilename.c_str(), m_tweak.copies - 1,
                           (m_tweak.cloneaxisX << 0) | (m_tweak.cloneaxisY << 1) | (m_tweak.cloneaxisZ << 2));

  const Renderer::Registry registry = Renderer::getRegistry();
  for(size_t i = 0; i < registry.size(); i++)
  {
    if(registry[i]->isAvailable())
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
  ImGui::SetNextWindowSize(ImVec2(280, 0), ImGuiCond_FirstUseEver);
  if(ImGui::Begin("NVIDIA " PROJECT_NAME, nullptr))
  {
    ImGui::PushItemWidth(120);
#if HAS_OPENGL
    ImGui::Text("gl and vk version");
#else
    ImGui::Text("vk only version");
#endif
    ImGui::Separator();
    if(!m_messageString.empty())
    {
      ImGui::Text("%s", m_messageString.c_str());
      ImGui::Separator();
    }


    m_ui.enumCombobox(GUI_RENDERER, "renderer", &m_tweak.renderer);
    m_ui.enumCombobox(GUI_VIEWPOINT, "viewpoint", &m_tweak.viewPoint);
    ImGuiH::InputIntClamped("min. task meshlets", &m_tweak.minTaskMeshlets, 0, 256, 1, 16, ImGuiInputTextFlags_EnterReturnsTrue);
    ImGui::Checkbox("(mesh) colorize by meshlet", &m_tweak.colorize);
    ImGui::Checkbox("use backface culling ", &m_tweak.useBackFaceCull);
    ImGui::Checkbox("show meshlet bboxes", &m_tweak.showBboxes);
    ImGui::Checkbox("show meshlet normals", &m_tweak.showNormals);
    ImGui::Checkbox("(show) culled", &m_tweak.showCulled);
    //ImGui::Checkbox("animate", &m_tweak.animate);
    ImGui::SliderFloat("fov", &m_tweak.fov, 1, 120, "%.0f");
    ImGui::Separator();
    ImGui::Checkbox("(mesh) use per-primitive culling ", &m_tweak.useMeshShaderCull);
    ImGui::Checkbox("use clipping planes", &m_tweak.useClipping);
    ImGui::SliderFloat3("clip position", m_tweak.clipPosition.vec_array, 0.01f, 1.01, "%.2f");
    ImGui::SliderFloat("(task) pixel cull", &m_tweak.pixelCull, 0.0f, 1.0f, "%.2f");
#if 0
      ImGui::Separator();
      ImGuiH::InputIntClamped("objectFrom", &m_tweak.objectFrom, 0, (int)m_scene.m_objects.size()-1);
      ImGuiH::InputIntClamped("objectNum", &m_tweak.objectNum, 0, (int)m_scene.m_objects.size());
      ImGui::InputInt("indexThreshold", &m_tweak.indexThreshold);
#endif
    ImGui::Separator();
    ImGuiH::InputIntClamped("meshlet vertices", &m_modelConfig.meshVertexCount, 32, 256, 32, 32, ImGuiInputTextFlags_EnterReturnsTrue);
    ImGuiH::InputIntClamped("meshlet primitives", &m_modelConfig.meshPrimitiveCount, 32, 256, 32, 32, ImGuiInputTextFlags_EnterReturnsTrue);
    ImGuiH::InputIntClamped("extra v4 attributes", &m_modelConfig.extraAttributes, 0, 7);
    ImGui::Checkbox("model fp16 attributes", &m_modelConfig.fp16);
    ImGuiH::InputIntClamped("model copies", &m_tweak.copies, 1, 256, 1, 10, ImGuiInputTextFlags_EnterReturnsTrue);
    ImGui::Separator();
    m_ui.enumCombobox(GUI_SUPERSAMPLE, "superresolution", &m_tweak.supersample);
    ImGui::Separator();

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
        m_statsCpuTime = info.cpu.average;
        m_statsGpuTime = info.gpu.average;
        m_lastFrameTime = time;
        m_frames        = -1;
      }

      m_frames++;

      float gpuTimeF = float(m_statsGpuTime);
      ImGui::Text("         Render GPU [ms]: %2.3f", gpuTimeF / 1000.0f);
      ImGui::Text("Original Index Size [MB]: %4zu", m_scene.m_iboSize / (1024 * 1024));
      ImGui::Text("       Meshlet Size [MB]: %4zu", m_scene.m_meshSize / (1024 * 1024));
    }
    ImGui::Checkbox("generate stats", &m_tweak.useStats);
    if(m_tweak.useStats)
    {
      ImGui::Separator();
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
    //m_ui.enumCombobox(GUI_MSAA, "msaa", &m_tweak.msaa);
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

  if(m_windowState.onPress(KEY_R) || m_tweak.useBackFaceCull != m_lastTweak.useBackFaceCull
     || m_tweak.useMeshShaderCull != m_lastTweak.useMeshShaderCull || m_tweak.useClipping != m_lastTweak.useClipping
     || m_tweak.useStats != m_lastTweak.useStats || m_modelConfig.extraAttributes != m_lastModelConfig.extraAttributes
     || m_modelConfig.meshPrimitiveCount != m_lastModelConfig.meshPrimitiveCount
     || m_modelConfig.meshVertexCount != m_lastModelConfig.meshVertexCount || m_shaderprepend != m_lastShaderPrepend
     || m_tweak.showBboxes != m_lastTweak.showBboxes || m_tweak.showNormals != m_lastTweak.showNormals
     || m_tweak.showCulled != m_lastTweak.showCulled)
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

  if(m_tweak.supersample != m_lastTweak.supersample || getVsync() != m_lastVsync)
  {
    m_lastVsync = getVsync();
    m_resources->initFramebuffer(width, height, m_tweak.supersample, getVsync());
  }

  bool sceneChanged = false;
  if(m_tweak.copies != m_lastTweak.copies || m_tweak.cloneaxisX != m_lastTweak.cloneaxisX
     || m_tweak.cloneaxisY != m_lastTweak.cloneaxisY || m_tweak.cloneaxisZ != m_lastTweak.cloneaxisZ
     || memcmp(&m_modelConfig, &m_lastModelConfig, sizeof(m_modelConfig)))
  {
    sceneChanged = true;
    m_resources->synchronize();
    deinitRenderer();
    m_resources->deinitScene();
    initScene(m_modelFilename.c_str(), m_tweak.copies - 1,
              (m_tweak.cloneaxisX << 0) | (m_tweak.cloneaxisY << 1) | (m_tweak.cloneaxisZ << 2));
    m_resources->initScene(m_scene);
  }

  if(sceneChanged || m_tweak.renderer != m_lastTweak.renderer || m_tweak.objectFrom != m_lastTweak.objectFrom
     || m_tweak.objectNum != m_lastTweak.objectNum || m_tweak.useClipping != m_lastTweak.useClipping
     || m_tweak.useStats != m_lastTweak.useStats || m_tweak.maxGroups != m_lastTweak.maxGroups
     || m_tweak.indexThreshold != m_lastTweak.indexThreshold || m_tweak.minTaskMeshlets != m_lastTweak.minTaskMeshlets)
  {
    m_resources->synchronize();
    initRenderer(m_tweak.renderer);
  }

  if(m_tweak.viewPoint != m_lastTweak.viewPoint)
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
    ;
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
  m_parameterList.add("meshshadercull", &m_tweak.useMeshShaderCull);
  m_parameterList.add("backfacecull", &m_tweak.useBackFaceCull);

  m_parameterList.add("showbbox", &m_tweak.showBboxes);
  m_parameterList.add("shownormals", &m_tweak.showNormals);
  m_parameterList.add("showculled", &m_tweak.showCulled);

  m_parameterList.add("stats", &m_tweak.useStats);

  m_parameterList.add("clipping", &m_tweak.useClipping);
  m_parameterList.add("clippos", m_tweak.clipPosition.vec_array, nullptr, 3);

  m_parameterList.add("message", &m_messageString);
}


bool Sample::validateConfig()
{
  if (m_modelFilename.empty())
  {
    std::vector<std::string> searchPaths;
    searchPaths.push_back("./");
    searchPaths.push_back(exePath() + PROJECT_RELDIRECTORY + "/");
    std::string configFile = nvh::findFile("worldcar_meshlet.cfg", searchPaths);
    if (!configFile.empty()) {
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
  NVPSystem system(argv[0], PROJECT_NAME);

  omp_set_num_threads(std::thread::hardware_concurrency());

  Sample sample;

  return sample.run(PROJECT_NAME, argc, argv, SAMPLE_SIZE_WIDTH, SAMPLE_SIZE_HEIGHT);
}
