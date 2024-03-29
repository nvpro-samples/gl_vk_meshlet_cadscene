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



#include "cadscene_gl.hpp"
#include "nvmeshlet_packbasic.hpp"
#include <cinttypes>
#include <nvh/nvprint.hpp>

//////////////////////////////////////////////////////////////////////////


static size_t alignedSize(size_t sz, size_t align)
{
  return ((sz + align - 1) / (align)) * align;
}

//////////////////////////////////////////////////////////////////////////

void GeometryMemoryGL::alloc(size_t vboSize, size_t aboSize, size_t iboSize, size_t meshSize, size_t meshIndicesSize, GeometryMemoryGL::Allocation& allocation)
{
  vboSize  = alignedSize(vboSize, m_vboAlignment);
  aboSize  = alignedSize(aboSize, m_aboAlignment);
  iboSize  = alignedSize(iboSize, m_alignment);
  meshSize = alignedSize(meshSize, m_alignment);
  meshIndicesSize = alignedSize(meshIndicesSize, m_alignment);

  if(m_chunks.empty() || getActiveChunk().vboSize + vboSize > m_maxVboChunk || getActiveChunk().aboSize + aboSize > m_maxVboChunk
     || getActiveChunk().iboSize + iboSize > m_maxIboChunk || getActiveChunk().meshSize + meshSize > m_maxChunk || getActiveChunk().meshIndicesSize + meshIndicesSize > m_maxMeshIndicesChunk)
  {
    finalize();
    Chunk chunk = {};
    m_chunks.push_back(chunk);
  }

  Chunk& chunk = getActiveChunk();

  allocation.chunkIndex = getActiveIndex();
  allocation.vboOffset  = chunk.vboSize;
  allocation.aboOffset  = chunk.aboSize;
  allocation.iboOffset  = chunk.iboSize;
  allocation.meshOffset = chunk.meshSize;
  allocation.meshIndicesOffset = chunk.meshIndicesSize;

  chunk.vboSize += vboSize;
  chunk.aboSize += aboSize;
  chunk.iboSize += iboSize;
  chunk.meshSize += meshSize;
  chunk.meshIndicesSize += meshIndicesSize;
}

void GeometryMemoryGL::finalize()
{
  if(m_chunks.empty())
  {
    return;
  }

  Chunk& chunk = getActiveChunk();

  glCreateBuffers(1, &chunk.vboGL);
  glNamedBufferStorage(chunk.vboGL, static_cast<GLsizeiptr>(chunk.vboSize), nullptr, GL_DYNAMIC_STORAGE_BIT);

  glCreateBuffers(1, &chunk.aboGL);
  glNamedBufferStorage(chunk.aboGL, static_cast<GLsizeiptr>(chunk.aboSize), nullptr, GL_DYNAMIC_STORAGE_BIT);

  glCreateBuffers(1, &chunk.iboGL);
  glNamedBufferStorage(chunk.iboGL, static_cast<GLsizeiptr>(chunk.iboSize), nullptr, GL_DYNAMIC_STORAGE_BIT);

  glCreateTextures(GL_TEXTURE_BUFFER, 1, &chunk.vboTEX);
  glTextureBuffer(chunk.vboTEX, m_fp16 ? GL_RGBA16F : GL_RGBA32F, chunk.vboGL);

  glCreateTextures(GL_TEXTURE_BUFFER, 1, &chunk.aboTEX);
  glTextureBuffer(chunk.aboTEX, m_fp16 ? GL_RGBA16F : GL_RGBA32F, chunk.aboGL);


  if(m_bindless)
  {
    glGetNamedBufferParameterui64vNV(chunk.vboGL, GL_BUFFER_GPU_ADDRESS_NV, &chunk.vboADDR);
    glMakeNamedBufferResidentNV(chunk.vboGL, GL_READ_ONLY);

    glGetNamedBufferParameterui64vNV(chunk.aboGL, GL_BUFFER_GPU_ADDRESS_NV, &chunk.aboADDR);
    glMakeNamedBufferResidentNV(chunk.aboGL, GL_READ_ONLY);

    glGetNamedBufferParameterui64vNV(chunk.iboGL, GL_BUFFER_GPU_ADDRESS_NV, &chunk.iboADDR);
    glMakeNamedBufferResidentNV(chunk.iboGL, GL_READ_ONLY);

    chunk.vboTEXADDR = glGetTextureHandleARB(chunk.vboTEX);
    glMakeTextureHandleResidentARB(chunk.vboTEXADDR);

    chunk.aboTEXADDR = glGetTextureHandleARB(chunk.aboTEX);
    glMakeTextureHandleResidentARB(chunk.aboTEXADDR);
  }

  if(chunk.meshSize)
  {
    // safety padding / minimum size
    chunk.meshSize = std::max(chunk.meshSize, size_t(16));
    chunk.meshIndicesSize += 16;

    glCreateBuffers(1, &chunk.meshGL);
    glNamedBufferStorage(chunk.meshGL, static_cast<GLsizeiptr>(chunk.meshSize), nullptr, GL_DYNAMIC_STORAGE_BIT);

    glCreateBuffers(1, &chunk.meshIndicesGL);
    glNamedBufferStorage(chunk.meshIndicesGL, static_cast<GLsizeiptr>(chunk.meshIndicesSize), nullptr, GL_DYNAMIC_STORAGE_BIT);

    if(m_bindless)
    {
      glGetNamedBufferParameterui64vNV(chunk.meshGL, GL_BUFFER_GPU_ADDRESS_NV, &chunk.meshADDR);
      glMakeNamedBufferResidentNV(chunk.meshGL, GL_READ_ONLY);

      glGetNamedBufferParameterui64vNV(chunk.meshIndicesGL, GL_BUFFER_GPU_ADDRESS_NV, &chunk.meshIndicesADDR);
      glMakeNamedBufferResidentNV(chunk.meshIndicesGL, GL_READ_ONLY);
    }
  }
}

void GeometryMemoryGL::init(size_t vboStride, size_t aboStride, size_t maxChunk, bool bindless, bool fp16)
{
  // buffer allocation
  // costs of entire model, provide offset into large buffers per geometry
  GLint tboAlign  = 1;
  GLint ssboAlign = 1;
  glGetIntegerv(GL_TEXTURE_BUFFER_OFFSET_ALIGNMENT, &tboAlign);
  glGetIntegerv(GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT, &ssboAlign);
  m_alignment = std::max(tboAlign, ssboAlign);

  // to keep vbo/abo "parallel" to each other, we need to use a common multiple
  // that means every offset of vbo/abo of the same sub-allocation can be expressed as "nth vertex" offset from the buffer
  size_t multiple = 1;
  while(true)
  {
    if(((multiple * vboStride) % m_alignment == 0) && ((multiple * aboStride) % m_alignment == 0))
    {
      break;
    }
    multiple++;
  }
  m_vboAlignment = multiple * vboStride;
  m_aboAlignment = multiple * aboStride;

  // buffer allocation
  // costs of entire model, provide offset into large buffers per geometry
  GLint tboSize = 0;
  glGetIntegerv(GL_MAX_TEXTURE_BUFFER_SIZE, &tboSize);

  const size_t vboMax  = size_t(tboSize) * sizeof(float) * 4;
  const size_t iboMax  = size_t(tboSize) * sizeof(uint16_t);
  const size_t meshMax = size_t(tboSize) * sizeof(uint16_t);

  m_maxVboChunk  = std::min(vboMax, maxChunk);
  m_maxIboChunk  = std::min(iboMax, maxChunk);
  m_maxMeshIndicesChunk = std::min(meshMax, maxChunk);

  m_maxChunk = maxChunk;
  m_bindless = bindless;
  m_fp16     = fp16;
}

void GeometryMemoryGL::deinit()
{
  for(auto & m_chunk : m_chunks)
  {
    if(m_bindless)
    {
      if(m_chunk.meshGL)
      {
        glMakeNamedBufferNonResidentNV(m_chunk.meshGL);
        glMakeNamedBufferNonResidentNV(m_chunk.meshIndicesGL);
      }


      glMakeTextureHandleNonResidentARB(m_chunk.vboTEXADDR);
      glMakeTextureHandleNonResidentARB(m_chunk.aboTEXADDR);

      glMakeNamedBufferNonResidentNV(m_chunk.vboGL);
      glMakeNamedBufferNonResidentNV(m_chunk.aboGL);
      glMakeNamedBufferNonResidentNV(m_chunk.iboGL);
    }

    if(m_chunk.meshGL)
    {
      glDeleteBuffers(1, &m_chunk.meshGL);
      glDeleteBuffers(1, &m_chunk.meshIndicesGL);
    }

    glDeleteTextures(1, &m_chunk.vboTEX);
    glDeleteTextures(1, &m_chunk.aboTEX);

    glDeleteBuffers(1, &m_chunk.vboGL);
    glDeleteBuffers(1, &m_chunk.aboGL);
    glDeleteBuffers(1, &m_chunk.iboGL);
  }

  m_chunks.clear();
}

//////////////////////////////////////////////////////////////////////////

void CadSceneGL::init(const CadScene& cadscene)
{
  m_geometry.resize(cadscene.m_geometry.size());

  {
    m_geometryMem.init(cadscene.getVertexSize(), cadscene.getVertexAttributeSize(), 128 * 1024 * 1024,
                       has_GL_NV_vertex_buffer_unified_memory != 0, cadscene.m_cfg.fp16);

    for(size_t i = 0; i < cadscene.m_geometry.size(); i++)
    {
      const CadScene::Geometry& cadgeom = cadscene.m_geometry[i];
      Geometry&                 geom    = m_geometry[i];

      m_geometryMem.alloc(cadgeom.vboSize, cadgeom.aboSize, cadgeom.iboSize, cadgeom.meshSize, cadgeom.meshIndicesSize, geom.mem);
    }

    m_geometryMem.finalize();

    LOGI("Size of vertex data: %11" PRId64 "\n", uint64_t(m_geometryMem.getVertexSize()))
    LOGI("Size of attrib data: %11" PRId64 "\n", uint64_t(m_geometryMem.getAttributeSize()))
    LOGI("Size of index data:  %11" PRId64 "\n", uint64_t(m_geometryMem.getIndexSize()))
    LOGI("Size of mesh data:   %11" PRId64 "\n", uint64_t(m_geometryMem.getMeshSize()))
    LOGI("Size of data:        %11" PRId64 "\n", uint64_t(m_geometryMem.getVertexSize() + m_geometryMem.getAttributeSize()
                                                          + m_geometryMem.getIndexSize() + m_geometryMem.getMeshSize()))
    LOGI("Chunks:              %11d\n", uint32_t(m_geometryMem.getChunkCount()))
  }

  for(size_t i = 0; i < cadscene.m_geometry.size(); i++)
  {
    const CadScene::Geometry& cadgeom = cadscene.m_geometry[i];
    Geometry&                 geom    = m_geometry[i];

    const GeometryMemoryGL::Chunk& chunk = m_geometryMem.getChunk(geom.mem);

    glNamedBufferSubData(chunk.vboGL, static_cast<GLintptr>(geom.mem.vboOffset), static_cast<GLsizeiptr>(cadgeom.vboSize), cadgeom.vboData);
    glNamedBufferSubData(chunk.aboGL, static_cast<GLintptr>(geom.mem.aboOffset), static_cast<GLsizeiptr>(cadgeom.aboSize), cadgeom.aboData);
    glNamedBufferSubData(chunk.iboGL, static_cast<GLintptr>(geom.mem.iboOffset), static_cast<GLsizeiptr>(cadgeom.iboSize), cadgeom.iboData);

    geom.vbo = nvgl::BufferBinding(chunk.vboGL, static_cast<GLintptr>(geom.mem.vboOffset), static_cast<GLsizeiptr>(cadgeom.vboSize), chunk.vboADDR);
    geom.abo = nvgl::BufferBinding(chunk.aboGL, static_cast<GLintptr>(geom.mem.aboOffset), static_cast<GLsizeiptr>(cadgeom.aboSize), chunk.aboADDR);
    geom.ibo = nvgl::BufferBinding(chunk.iboGL, static_cast<GLintptr>(geom.mem.iboOffset), static_cast<GLsizeiptr>(cadgeom.iboSize), chunk.iboADDR);

    GLintptr descOffset = static_cast<GLintptr>(geom.mem.meshOffset);
    GLintptr primOffset = static_cast<GLintptr>(geom.mem.meshIndicesOffset);

    glNamedBufferSubData(chunk.meshGL, descOffset, static_cast<GLsizeiptr>(cadgeom.meshlet.descSize), cadgeom.meshlet.descData);
    glNamedBufferSubData(chunk.meshIndicesGL, primOffset, static_cast<GLsizeiptr>(cadgeom.meshlet.primSize), cadgeom.meshlet.primData);

    geom.topoMeshlet = nvgl::BufferBinding(chunk.meshGL, descOffset, static_cast<GLsizeiptr>(cadgeom.meshlet.descSize), chunk.meshADDR);
    geom.topoPrim    = nvgl::BufferBinding(chunk.meshIndicesGL, primOffset, static_cast<GLsizeiptr>(cadgeom.meshlet.primSize), chunk.meshIndicesADDR);
  }

  m_buffers.materials.create(sizeof(CadScene::Material) * cadscene.m_materials.size(), cadscene.m_materials.data(), 0, 0);
  m_buffers.matrices.create(sizeof(CadScene::MatrixNode) * cadscene.m_matrices.size(), cadscene.m_matrices.data(), 0, 0);
}

void CadSceneGL::deinit()
{
  if(m_geometry.empty())
    return;

  m_buffers.matrices.destroy();
  m_buffers.materials.destroy();

  m_geometryMem.deinit();

  m_geometry.clear();
}
