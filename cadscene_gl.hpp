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
    size_t meshIndicesOffset;
  };

  struct Chunk
  {
    GLuint vboGL;
    GLuint aboGL;
    GLuint iboGL;
    GLuint meshGL = 0;
    GLuint meshIndicesGL = 0;

    GLuint vboTEX;
    GLuint aboTEX;

    size_t vboSize;
    size_t aboSize;
    size_t iboSize;
    size_t meshSize;
    size_t meshIndicesSize;

    uint64_t vboADDR;
    uint64_t aboADDR;
    uint64_t iboADDR;
    uint64_t meshADDR;
    uint64_t meshIndicesADDR;

    uint64_t vboTEXADDR;
    uint64_t aboTEXADDR;
  };

  void init(size_t vboStride, size_t aboStride, size_t maxChunk, bool bindless, bool fp16);
  void deinit();
  void alloc(size_t vboSize, size_t aboSize, size_t iboSize, size_t meshSize, size_t meshIndicesSize, Allocation& allocation);
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
      size += m_chunks[i].meshSize + m_chunks[i].meshIndicesSize;
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
  size_t m_maxMeshIndicesChunk;
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
