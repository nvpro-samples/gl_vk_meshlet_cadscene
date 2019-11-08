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


#include "cadscene_gl.hpp"
#include "nvmeshlet_builder.hpp"
#include <inttypes.h>
#include <nvgl/glsltypes_gl.hpp>
#include <nvh/nvprint.hpp>

#include "common.h"


//////////////////////////////////////////////////////////////////////////


static size_t alignedSize(size_t sz, size_t align)
{
  return ((sz + align - 1) / (align)) * align;
}

//////////////////////////////////////////////////////////////////////////

void GeometryMemoryGL::alloc(size_t vboSize, size_t aboSize, size_t iboSize, size_t meshSize, GeometryMemoryGL::Allocation& allocation)
{
  vboSize  = alignedSize(vboSize, m_vboAlignment);
  aboSize  = alignedSize(aboSize, m_aboAlignment);
  iboSize  = alignedSize(iboSize, m_alignment);
  meshSize = alignedSize(meshSize, m_alignment);

  if(m_chunks.empty() || getActiveChunk().vboSize + vboSize > m_maxVboChunk || getActiveChunk().aboSize + aboSize > m_maxVboChunk
     || getActiveChunk().iboSize + iboSize > m_maxIboChunk || getActiveChunk().meshSize + meshSize > m_maxMeshChunk)
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

  chunk.vboSize += vboSize;
  chunk.aboSize += aboSize;
  chunk.iboSize += iboSize;
  chunk.meshSize += meshSize;
}

void GeometryMemoryGL::finalize()
{
  if(m_chunks.empty())
  {
    return;
  }

  Chunk& chunk = getActiveChunk();
  glCreateBuffers(1, &chunk.vboGL);
  glNamedBufferStorage(chunk.vboGL, chunk.vboSize, 0, GL_DYNAMIC_STORAGE_BIT);

  glCreateBuffers(1, &chunk.aboGL);
  glNamedBufferStorage(chunk.aboGL, chunk.aboSize, 0, GL_DYNAMIC_STORAGE_BIT);

  glCreateBuffers(1, &chunk.iboGL);
  glNamedBufferStorage(chunk.iboGL, chunk.iboSize, 0, GL_DYNAMIC_STORAGE_BIT);

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
    glCreateBuffers(1, &chunk.meshGL);
    glNamedBufferStorage(chunk.meshGL, chunk.meshSize, 0, GL_DYNAMIC_STORAGE_BIT);

    glCreateTextures(GL_TEXTURE_BUFFER, 1, &chunk.meshVertex16TEX);
    glTextureBuffer(chunk.meshVertex16TEX, GL_R16UI, chunk.meshGL);

    glCreateTextures(GL_TEXTURE_BUFFER, 1, &chunk.meshVertex32TEX);
    glTextureBuffer(chunk.meshVertex32TEX, GL_R32UI, chunk.meshGL);

    if(m_bindless)
    {
      glGetNamedBufferParameterui64vNV(chunk.meshGL, GL_BUFFER_GPU_ADDRESS_NV, &chunk.meshADDR);
      glMakeNamedBufferResidentNV(chunk.meshGL, GL_READ_ONLY);

      chunk.meshVertex16TEXADDR = glGetTextureHandleARB(chunk.meshVertex16TEX);
      chunk.meshVertex32TEXADDR = glGetTextureHandleARB(chunk.meshVertex32TEX);
      glMakeTextureHandleResidentARB(chunk.meshVertex16TEXADDR);
      glMakeTextureHandleResidentARB(chunk.meshVertex32TEXADDR);
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
  m_maxMeshChunk = std::min(meshMax, maxChunk);

  m_maxChunk = maxChunk;
  m_bindless = bindless;
  m_fp16     = fp16;
}

void GeometryMemoryGL::deinit()
{
  for(size_t i = 0; i < m_chunks.size(); i++)
  {
    if(m_bindless)
    {
      if(m_chunks[i].meshGL)
      {
        glMakeTextureHandleNonResidentARB(m_chunks[i].meshVertex16TEXADDR);
        glMakeTextureHandleNonResidentARB(m_chunks[i].meshVertex32TEXADDR);
        glMakeNamedBufferNonResidentNV(m_chunks[i].meshGL);
      }


      glMakeTextureHandleNonResidentARB(m_chunks[i].vboTEXADDR);
      glMakeTextureHandleNonResidentARB(m_chunks[i].aboTEXADDR);

      glMakeNamedBufferNonResidentNV(m_chunks[i].vboGL);
      glMakeNamedBufferNonResidentNV(m_chunks[i].aboGL);
      glMakeNamedBufferNonResidentNV(m_chunks[i].iboGL);
    }

    if(m_chunks[i].meshGL)
    {
      glDeleteTextures(1, &m_chunks[i].meshVertex16TEX);
      glDeleteTextures(1, &m_chunks[i].meshVertex32TEX);
      glDeleteBuffers(1, &m_chunks[i].meshGL);
    }

    glDeleteTextures(1, &m_chunks[i].vboTEX);
    glDeleteTextures(1, &m_chunks[i].aboTEX);

    glDeleteBuffers(1, &m_chunks[i].vboGL);
    glDeleteBuffers(1, &m_chunks[i].aboGL);
    glDeleteBuffers(1, &m_chunks[i].iboGL);
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

      m_geometryMem.alloc(cadgeom.vboSize, cadgeom.aboSize, cadgeom.iboSize, cadgeom.meshSize, geom.mem);
    }

    m_geometryMem.finalize();

    LOGI("Size of vertex data: %11" PRId64 "\n", uint64_t(m_geometryMem.getVertexSize()));
    LOGI("Size of attrib data: %11" PRId64 "\n", uint64_t(m_geometryMem.getAttributeSize()));
    LOGI("Size of index data:  %11" PRId64 "\n", uint64_t(m_geometryMem.getIndexSize()));
    LOGI("Size of mesh data:   %11" PRId64 "\n", uint64_t(m_geometryMem.getMeshSize()));
    LOGI("Size of data:        %11" PRId64 "\n", uint64_t(m_geometryMem.getVertexSize() + m_geometryMem.getAttributeSize()
                                                          + m_geometryMem.getIndexSize() + m_geometryMem.getMeshSize()));
    LOGI("Chunks:              %11d\n", uint32_t(m_geometryMem.getChunkCount()));
  }

  for(size_t i = 0; i < cadscene.m_geometry.size(); i++)
  {
    const CadScene::Geometry& cadgeom = cadscene.m_geometry[i];
    Geometry&                 geom    = m_geometry[i];

    const GeometryMemoryGL::Chunk& chunk = m_geometryMem.getChunk(geom.mem);

    glNamedBufferSubData(chunk.vboGL, geom.mem.vboOffset, cadgeom.vboSize, cadgeom.vboData);
    glNamedBufferSubData(chunk.aboGL, geom.mem.aboOffset, cadgeom.aboSize, cadgeom.aboData);
    glNamedBufferSubData(chunk.iboGL, geom.mem.iboOffset, cadgeom.iboSize, cadgeom.iboData);

    geom.vbo = nvgl::BufferBinding(chunk.vboGL, geom.mem.vboOffset, cadgeom.vboSize, chunk.vboADDR);
    geom.abo = nvgl::BufferBinding(chunk.aboGL, geom.mem.aboOffset, cadgeom.aboSize, chunk.aboADDR);
    geom.ibo = nvgl::BufferBinding(chunk.iboGL, geom.mem.iboOffset, cadgeom.iboSize, chunk.iboADDR);

    GLintptr descOffset = geom.mem.meshOffset;
    GLintptr primOffset = geom.mem.meshOffset + NVMeshlet::computeCommonAlignedSize(cadgeom.meshlet.descSize);
    GLintptr vertOffset = geom.mem.meshOffset + NVMeshlet::computeCommonAlignedSize(cadgeom.meshlet.descSize)
                          + NVMeshlet::computeCommonAlignedSize(cadgeom.meshlet.primSize);

    glNamedBufferSubData(chunk.meshGL, descOffset, cadgeom.meshlet.descSize, cadgeom.meshlet.descData);
    glNamedBufferSubData(chunk.meshGL, primOffset, cadgeom.meshlet.primSize, cadgeom.meshlet.primData);
    glNamedBufferSubData(chunk.meshGL, vertOffset, cadgeom.meshlet.vertSize, cadgeom.meshlet.vertData);

    geom.topoMeshlet = nvgl::BufferBinding(chunk.meshGL, descOffset, cadgeom.meshlet.descSize, chunk.meshADDR);
    geom.topoPrim    = nvgl::BufferBinding(chunk.meshGL, primOffset, cadgeom.meshlet.primSize, chunk.meshADDR);
    geom.topoVert    = nvgl::BufferBinding(chunk.meshGL, vertOffset, cadgeom.meshlet.vertSize, chunk.meshADDR);

#if USE_PER_GEOMETRY_VIEWS
    geom.vboTEX.create(chunk.vboGL, geom.mem.vboOffset, cadgeom.vboSize, cadscene.m_cfg.fp16 ? GL_RGBA16F : GL_RGBA32F);
    geom.aboTEX.create(chunk.aboGL, geom.mem.aboOffset, cadgeom.aboSize, cadscene.m_cfg.fp16 ? GL_RGBA16F : GL_RGBA32F);
    geom.vertTEX.create(chunk.meshGL, vertOffset, cadgeom.meshlet.vertSize, cadgeom.useShorts ? GL_R16UI : GL_R32UI);
#endif
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

#if USE_PER_GEOMETRY_VIEWS
  for(size_t i = 0; i < m_geometry.size(); i++)
  {
    m_geometry[i].vboTEX.destroy();
    m_geometry[i].aboTEX.destroy();
    m_geometry[i].vertTEX.destroy();
  }
#endif

  m_geometryMem.deinit();

  m_geometry.clear();
}
