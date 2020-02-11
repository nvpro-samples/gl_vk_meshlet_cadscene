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

#pragma once

#include "cadscene.hpp"
#include <include_gl.h>
#include <nvgl/base_gl.hpp>


class GeometryMemoryGL
{
public:
  typedef size_t Index;

  struct Allocation
  {
    Index  chunkIndex;
    size_t vboOffset;
    size_t aboOffset;
    size_t iboOffset;
    size_t meshOffset;
  };

  struct Chunk
  {
    GLuint vboGL;
    GLuint aboGL;
    GLuint iboGL;
    GLuint meshGL = 0;

    GLuint vboTEX;
    GLuint aboTEX;
    GLuint meshVertex32TEX;
    GLuint meshVertex16TEX;

    size_t vboSize;
    size_t aboSize;
    size_t iboSize;
    size_t meshSize;

    uint64_t vboADDR;
    uint64_t aboADDR;
    uint64_t iboADDR;
    uint64_t meshADDR;

    uint64_t vboTEXADDR;
    uint64_t aboTEXADDR;
    uint64_t meshVertex32TEXADDR;
    uint64_t meshVertex16TEXADDR;
  };

  void init(size_t vboStride, size_t aboStride, size_t maxChunk, bool bindless, bool fp16);
  void deinit();
  void alloc(size_t vboSize, size_t aboSize, size_t iboSize, size_t meshSize, Allocation& allocation);
  void finalize();

  size_t getVertexSize() const
  {
    size_t size = 0;
    for(size_t i = 0; i < m_chunks.size(); i++)
    {
      size += m_chunks[i].vboSize;
    }
    return size;
  }

  size_t getAttributeSize() const
  {
    size_t size = 0;
    for(size_t i = 0; i < m_chunks.size(); i++)
    {
      size += m_chunks[i].aboSize;
    }
    return size;
  }

  size_t getIndexSize() const
  {
    size_t size = 0;
    for(size_t i = 0; i < m_chunks.size(); i++)
    {
      size += m_chunks[i].iboSize;
    }
    return size;
  }

  size_t getMeshSize() const
  {
    size_t size = 0;
    for(size_t i = 0; i < m_chunks.size(); i++)
    {
      size += m_chunks[i].meshSize;
    }
    return size;
  }

  const Chunk& getChunk(const Allocation& allocation) const { return m_chunks[allocation.chunkIndex]; }

  const Chunk& getChunk(Index index) const { return m_chunks[index]; }

  size_t getChunkCount() const { return m_chunks.size(); }

private:
  size_t m_alignment;
  size_t m_vboAlignment;
  size_t m_aboAlignment;
  size_t m_maxChunk;
  size_t m_maxVboChunk;
  size_t m_maxIboChunk;
  size_t m_maxMeshChunk;
  bool   m_bindless;
  bool   m_fp16;

  std::vector<Chunk> m_chunks;

  Index getActiveIndex() { return (m_chunks.size() - 1); }

  Chunk& getActiveChunk()
  {
    assert(!m_chunks.empty());
    return m_chunks[getActiveIndex()];
  }
};

class CadSceneGL
{
public:
  struct Geometry
  {
    GeometryMemoryGL::Allocation mem;

    nvgl::BufferBinding vbo;
    nvgl::BufferBinding abo;
    nvgl::BufferBinding ibo;

    nvgl::BufferBinding topoMeshlet;
    nvgl::BufferBinding topoPrim;
    nvgl::BufferBinding topoVert;

#if USE_PER_GEOMETRY_VIEWS
    nvgl::TextureBuffer vboTEX;
    nvgl::TextureBuffer aboTEX;
    nvgl::TextureBuffer vertTEX;
#endif
  };

  struct GeometryUbo
  {
    uint8_t uboData[256];
  };

  struct Buffers
  {
    nvgl::Buffer matrices;
    nvgl::Buffer materials;
  };

  Buffers               m_buffers;
  std::vector<Geometry> m_geometry;
  GeometryMemoryGL      m_geometryMem;


  void init(const CadScene& cadscene);
  void deinit();
};
