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

#include <nvvk/buffers_vk.hpp>
#include <nvvk/commands_vk.hpp>
#include <nvvk/memorymanagement_vk.hpp>

// ScopeStaging handles uploads and other staging operations.
// not efficient because it blocks/syncs operations

struct ScopeStaging
{

  ScopeStaging(VkDevice device, VkPhysicalDevice physical, VkQueue queue, uint32_t queueFamily, VkDeviceSize size = 128 * 1024 * 1024)
      : staging(device, physical, size)
      , cmdPool(device, queue, queueFamily)
      , cmd(VK_NULL_HANDLE)
  {
  }

  VkCommandBuffer          cmd;
  nvvk::ScopeStagingBuffer staging;
  nvvk::ScopeSubmitCmdPool cmdPool;

  VkCommandBuffer getCmd()
  {
    cmd = cmd ? cmd : cmdPool.begin();
    return cmd;
  }

  void submit()
  {
    if(cmd)
    {
      cmdPool.end(cmd);
      cmd = VK_NULL_HANDLE;
    }
  }

  void upload(const VkDescriptorBufferInfo& binding, const void* data)
  {
    if(cmd && (data == nullptr || staging.doesNotFit(binding.range)))
    {
      submit();
      staging.flush();
    }
    if(data && binding.range)
    {
      staging.cmdToBuffer(getCmd(), binding.buffer, binding.offset, binding.range, data);
    }
  }
};


// GeometryMemoryVK manages vbo/ibo etc. in chunks
// allows to reduce number of bindings and be more memory efficient

struct GeometryMemoryVK
{
  typedef size_t Index;


  struct Allocation
  {
    Index        chunkIndex;
    VkDeviceSize vboOffset;
    VkDeviceSize aboOffset;
    VkDeviceSize iboOffset;
    VkDeviceSize meshOffset;
  };

  struct Chunk
  {
    VkBuffer vbo;
    VkBuffer ibo;
    VkBuffer abo;
    VkBuffer mesh;

    VkDescriptorBufferInfo meshInfo;
    VkBufferView           vboView;
    VkBufferView           aboView;
    VkBufferView           vert16View;
    VkBufferView           vert32View;

    VkDeviceSize vboSize;
    VkDeviceSize aboSize;
    VkDeviceSize iboSize;
    VkDeviceSize meshSize;

    nvvk::AllocationID vboAID;
    nvvk::AllocationID aboAID;
    nvvk::AllocationID iboAID;
    nvvk::AllocationID meshAID;
  };


  VkDevice                     m_device = VK_NULL_HANDLE;
  nvvk::DeviceMemoryAllocator* m_memoryAllocator;
  std::vector<Chunk>           m_chunks;
  bool                         m_fp16 = false;

  void init(VkDevice                     device,
            VkPhysicalDevice             physicalDevice,
            nvvk::DeviceMemoryAllocator* memoryAllocator,
            VkDeviceSize                 vboStride,
            VkDeviceSize                 aboStride,
            VkDeviceSize                 maxChunk);
  void deinit();
  void alloc(VkDeviceSize vboSize, VkDeviceSize aboSize, VkDeviceSize iboSize, VkDeviceSize meshSize, Allocation& allocation);
  void finalize();

  const Chunk& getChunk(const Allocation& allocation) const { return m_chunks[allocation.chunkIndex]; }

  const Chunk& getChunk(Index index) const { return m_chunks[index]; }

  VkDeviceSize getVertexSize() const
  {
    VkDeviceSize size = 0;
    for(size_t i = 0; i < m_chunks.size(); i++)
    {
      size += m_chunks[i].vboSize;
    }
    return size;
  }

  VkDeviceSize getAttributeSize() const
  {
    VkDeviceSize size = 0;
    for(size_t i = 0; i < m_chunks.size(); i++)
    {
      size += m_chunks[i].aboSize;
    }
    return size;
  }

  VkDeviceSize getIndexSize() const
  {
    VkDeviceSize size = 0;
    for(size_t i = 0; i < m_chunks.size(); i++)
    {
      size += m_chunks[i].iboSize;
    }
    return size;
  }

  VkDeviceSize getMeshSize() const
  {
    VkDeviceSize size = 0;
    for(size_t i = 0; i < m_chunks.size(); i++)
    {
      size += m_chunks[i].meshSize;
    }
    return size;
  }

  VkDeviceSize getChunkCount() const { return m_chunks.size(); }

private:
  VkDeviceSize m_alignment;
  VkDeviceSize m_vboAlignment;
  VkDeviceSize m_aboAlignment;
  VkDeviceSize m_maxVboChunk;
  VkDeviceSize m_maxIboChunk;
  VkDeviceSize m_maxMeshChunk;

  Index getActiveIndex() { return (m_chunks.size() - 1); }

  Chunk& getActiveChunk()
  {
    assert(!m_chunks.empty());
    return m_chunks[getActiveIndex()];
  }
};


class CadSceneVK
{
public:
  struct Geometry
  {
    GeometryMemoryVK::Allocation allocation;

    VkDescriptorBufferInfo vbo;
    VkDescriptorBufferInfo abo;
    VkDescriptorBufferInfo ibo;

    VkDescriptorBufferInfo meshletDesc;
    VkDescriptorBufferInfo meshletPrim;
    VkDescriptorBufferInfo meshletVert;

#if USE_PER_GEOMETRY_VIEWS
    VkBufferView vboView;
    VkBufferView aboView;
    VkBufferView vertView;
#endif
  };

  struct Buffers
  {
    VkBuffer materials = VK_NULL_HANDLE;
    VkBuffer matrices  = VK_NULL_HANDLE;

    nvvk::AllocationID materialsAID;
    nvvk::AllocationID matricesAID;
  };

  struct Infos
  {
    VkDescriptorBufferInfo materialsSingle;
    VkDescriptorBufferInfo materials;
    VkDescriptorBufferInfo matricesSingle;
    VkDescriptorBufferInfo matrices;
  };


  VkDevice                    m_device = VK_NULL_HANDLE;
  nvvk::DeviceMemoryAllocator m_memAllocator;

  Buffers m_buffers;
  Infos   m_infos;

  std::vector<Geometry> m_geometry;
  GeometryMemoryVK      m_geometryMem;


  void init(const CadScene& cadscene, VkDevice device, VkPhysicalDevice physicalDevice, VkQueue queue, uint32_t queueFamilyIndex);
  void deinit();
};
