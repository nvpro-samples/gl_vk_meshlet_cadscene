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

#include "nvmeshlet_builder.hpp"

#include "cadscene_vk.hpp"

#include <algorithm>
#include <inttypes.h>
#include <nvh/nvprint.hpp>


static inline VkDeviceSize alignedSize(VkDeviceSize sz, VkDeviceSize align)
{
  return ((sz + align - 1) / (align)) * align;
}


void GeometryMemoryVK::init(VkDevice                     device,
                            VkPhysicalDevice             physicalDevice,
                            nvvk::DeviceMemoryAllocator* memoryAllocator,
                            VkDeviceSize                 vboStride,
                            VkDeviceSize                 aboStride,
                            VkDeviceSize                 maxChunk)
{
  m_device          = device;
  m_memoryAllocator = memoryAllocator;

  VkPhysicalDeviceProperties properties;
  vkGetPhysicalDeviceProperties(physicalDevice, &properties);
  VkPhysicalDeviceLimits& limits = properties.limits;

  m_alignment = std::max(limits.minTexelBufferOffsetAlignment, limits.minStorageBufferOffsetAlignment);
  // to keep vbo/abo "parallel" to each other, we need to use a common multiple
  // that means every offset of vbo/abo of the same sub-allocation can be expressed as "nth vertex" offset from the buffer
  VkDeviceSize multiple = 1;
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
  VkDeviceSize tboSize = limits.maxTexelBufferElements;

  const VkDeviceSize vboMax  = VkDeviceSize(tboSize) * sizeof(float) * 4;
  const VkDeviceSize iboMax  = VkDeviceSize(tboSize) * sizeof(uint16_t);
  const VkDeviceSize meshMax = VkDeviceSize(tboSize) * sizeof(uint16_t);

  m_maxVboChunk  = std::min(vboMax, maxChunk);
  m_maxIboChunk  = std::min(iboMax, maxChunk);
  m_maxMeshChunk = std::min(meshMax, maxChunk);
}

void GeometryMemoryVK::deinit()
{
  for(size_t i = 0; i < m_chunks.size(); i++)
  {
    const Chunk& chunk = getChunk(i);

    vkDestroyBufferView(m_device, chunk.vboView, nullptr);
    vkDestroyBufferView(m_device, chunk.aboView, nullptr);
    vkDestroyBufferView(m_device, chunk.vert16View, nullptr);
    vkDestroyBufferView(m_device, chunk.vert32View, nullptr);

    vkDestroyBuffer(m_device, chunk.vbo, nullptr);
    vkDestroyBuffer(m_device, chunk.abo, nullptr);
    vkDestroyBuffer(m_device, chunk.ibo, nullptr);
    vkDestroyBuffer(m_device, chunk.mesh, nullptr);

    m_memoryAllocator->free(chunk.vboAID);
    m_memoryAllocator->free(chunk.aboAID);
    m_memoryAllocator->free(chunk.iboAID);
    m_memoryAllocator->free(chunk.meshAID);
  }
  m_chunks          = std::vector<Chunk>();
  m_device          = nullptr;
  m_memoryAllocator = nullptr;
}

void GeometryMemoryVK::alloc(VkDeviceSize vboSize, VkDeviceSize aboSize, VkDeviceSize iboSize, VkDeviceSize meshSize, Allocation& allocation)
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

void GeometryMemoryVK::finalize()
{
  if(m_chunks.empty())
  {
    return;
  }

  Chunk& chunk = getActiveChunk();

  VkBufferUsageFlags flags = VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT;

  chunk.vbo = m_memoryAllocator->createBuffer(chunk.vboSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | flags, chunk.vboAID);
  chunk.abo = m_memoryAllocator->createBuffer(chunk.aboSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | flags, chunk.aboAID);
  chunk.ibo = m_memoryAllocator->createBuffer(chunk.iboSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT | flags, chunk.iboAID);
  chunk.mesh = m_memoryAllocator->createBuffer(chunk.meshSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | flags, chunk.meshAID);

  chunk.meshInfo   = {chunk.mesh, 0, chunk.meshSize};
  chunk.vboView    = nvvk::createBufferView(m_device, chunk.vbo,
                                         m_fp16 ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_R32G32B32A32_SFLOAT, chunk.vboSize);
  chunk.aboView    = nvvk::createBufferView(m_device, chunk.abo,
                                         m_fp16 ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_R32G32B32A32_SFLOAT, chunk.aboSize);
  chunk.vert16View = nvvk::createBufferView(m_device, chunk.mesh, VK_FORMAT_R16_UINT, chunk.meshSize);
  chunk.vert32View = nvvk::createBufferView(m_device, chunk.mesh, VK_FORMAT_R32_UINT, chunk.meshSize);
}

void CadSceneVK::init(const CadScene& cadscene, VkDevice device, VkPhysicalDevice physicalDevice, VkQueue queue, uint32_t queueFamilyIndex)
{
  m_device = device;

  m_memAllocator.init(m_device, physicalDevice, 1024 * 1024 * 256);

  m_geometry.resize(cadscene.m_geometry.size(), {0});

  if(m_geometry.empty())
    return;

  {
    // allocation phase
    m_geometryMem.init(device, physicalDevice, &m_memAllocator, cadscene.getVertexSize(),
                       cadscene.getVertexAttributeSize(), 512 * 1024 * 1024);
    m_geometryMem.m_fp16 = cadscene.m_cfg.fp16;

    for(size_t g = 0; g < cadscene.m_geometry.size(); g++)
    {
      const CadScene::Geometry& cadgeom = cadscene.m_geometry[g];
      Geometry&                 geom    = m_geometry[g];

      m_geometryMem.alloc(cadgeom.vboSize, cadgeom.aboSize, cadgeom.iboSize, cadgeom.meshSize, geom.allocation);
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

  {
    VkDeviceSize allocatedSize, usedSize;
    m_memAllocator.getUtilization(allocatedSize, usedSize);
    LOGI("scene geometry: used %d KB allocated %d KB\n", usedSize / 1024, allocatedSize / 1024);
  }

  ScopeStaging staging(device, physicalDevice, queue, queueFamilyIndex);

  for(size_t g = 0; g < cadscene.m_geometry.size(); g++)
  {
    const CadScene::Geometry&      cadgeom = cadscene.m_geometry[g];
    Geometry&                      geom    = m_geometry[g];
    const GeometryMemoryVK::Chunk& chunk   = m_geometryMem.getChunk(geom.allocation);

    // upload and assignment phase
    geom.vbo.buffer = chunk.vbo;
    geom.vbo.offset = geom.allocation.vboOffset;
    geom.vbo.range  = cadgeom.vboSize;
    staging.upload(geom.vbo, cadgeom.vboData);

    geom.abo.buffer = chunk.abo;
    geom.abo.offset = geom.allocation.aboOffset;
    geom.abo.range  = cadgeom.aboSize;
    staging.upload(geom.abo, cadgeom.aboData);

    geom.ibo.buffer = chunk.ibo;
    geom.ibo.offset = geom.allocation.iboOffset;
    geom.ibo.range  = cadgeom.iboSize;
    staging.upload(geom.ibo, cadgeom.iboData);

    if(cadgeom.meshSize)
    {
      geom.meshletDesc.buffer = chunk.mesh;
      geom.meshletDesc.offset = geom.allocation.meshOffset;
      geom.meshletDesc.range  = cadgeom.meshlet.descSize;
      staging.upload(geom.meshletDesc, cadgeom.meshlet.descData);

      geom.meshletPrim.buffer = chunk.mesh;
      geom.meshletPrim.offset = geom.allocation.meshOffset + NVMeshlet::computeCommonAlignedSize(cadgeom.meshlet.descSize);
      geom.meshletPrim.range = cadgeom.meshlet.primSize;
      staging.upload(geom.meshletPrim, cadgeom.meshlet.primData);

      geom.meshletVert.buffer = chunk.mesh;
      geom.meshletVert.offset = geom.allocation.meshOffset + NVMeshlet::computeCommonAlignedSize(cadgeom.meshlet.descSize)
                                + NVMeshlet::computeCommonAlignedSize(cadgeom.meshlet.primSize);
      geom.meshletVert.range = cadgeom.meshlet.vertSize;
      staging.upload(geom.meshletVert, cadgeom.meshlet.vertData);

#if USE_PER_GEOMETRY_VIEWS
      // views
      geom.vboView = nvvk::createBufferView(device, geom.vbo, cadscene.m_cfg.fp16 ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_R32G32B32A32_SFLOAT);
      geom.aboView = nvvk::createBufferView(device, geom.abo, cadscene.m_cfg.fp16 ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_R32G32B32A32_SFLOAT);
      geom.vertView = nvvk::createBufferView(device, geom.meshletVert, cadgeom.useShorts ? VK_FORMAT_R16_UINT : VK_FORMAT_R32_UINT);
#endif
    }
  }

  VkBufferUsageFlags bufferUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;

  m_buffers.materials = m_memAllocator.createBuffer(cadscene.m_materials.size() * sizeof(CadScene::Material),
                                                    bufferUsage, m_buffers.materialsAID);
  m_buffers.matrices  = m_memAllocator.createBuffer(cadscene.m_matrices.size() * sizeof(CadScene::MatrixNode),
                                                   bufferUsage, m_buffers.matricesAID);

  m_infos.materialsSingle = {m_buffers.materials, 0, sizeof(CadScene::Material)};
  m_infos.materials       = {m_buffers.materials, 0, cadscene.m_materials.size() * sizeof(CadScene::Material)};
  m_infos.matricesSingle  = {m_buffers.matrices, 0, sizeof(CadScene::MatrixNode)};
  m_infos.matrices        = {m_buffers.matrices, 0, cadscene.m_matrices.size() * sizeof(CadScene::MatrixNode)};

  staging.upload(m_infos.materials, cadscene.m_materials.data());
  staging.upload(m_infos.matrices, cadscene.m_matrices.data());

  staging.upload({}, nullptr);
}

void CadSceneVK::deinit()
{
#if USE_PER_GEOMETRY_VIEWS
  for(auto it = m_geometry.begin(); it != m_geometry.end(); it++)
  {
    if(it->aboView)
    {
      vkDestroyBufferView(m_device, it->vboView, NULL);
      vkDestroyBufferView(m_device, it->aboView, NULL);
      vkDestroyBufferView(m_device, it->vertView, NULL);
    }
  }
#endif

  vkDestroyBuffer(m_device, m_buffers.materials, nullptr);
  vkDestroyBuffer(m_device, m_buffers.matrices, nullptr);

  m_memAllocator.free(m_buffers.matricesAID);
  m_memAllocator.free(m_buffers.materialsAID);
  m_geometry.clear();
  m_geometryMem.deinit();
  m_memAllocator.deinit();
}
