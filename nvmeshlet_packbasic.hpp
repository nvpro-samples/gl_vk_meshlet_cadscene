/* Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef _NV_MESHLET_PACKBASIC_H__
#define _NV_MESHLET_PACKBASIC_H__

#include "nvmeshlet_builder.hpp"

namespace NVMeshlet {

static const uint32_t PACKBASIC_ALIGN = 16;
// how many indices are fetched per thread, 8 or 4
static const uint32_t PACKBASIC_PRIMITIVE_INDICES_PER_FETCH = 8;

typedef uint32_t PackBasicType;

struct MeshletPackBasicDesc
{
  //
  // Bitfield layout :
  //
  //   Field.X    | Bits | Content
  //  ------------|:----:|----------------------------------------------
  //  bboxMinX    | 8    | bounding box coord relative to object bbox
  //  bboxMinY    | 8    | UNORM8
  //  bboxMinZ    | 8    |
  //  vertexMax   | 8    | number of vertex indices - 1 in the meshlet
  //  ------------|:----:|----------------------------------------------
  //   Field.Y    |      |
  //  ------------|:----:|----------------------------------------------
  //  bboxMaxX    | 8    | bounding box coord relative to object bbox
  //  bboxMaxY    | 8    | UNORM8
  //  bboxMaxZ    | 8    |
  //  primMax     | 8    | number of primitives - 1 in the meshlet
  //  ------------|:----:|----------------------------------------------
  //   Field.Z    |      |
  //  ------------|:----:|----------------------------------------------
  //  coneOctX    | 8    | octant coordinate for cone normal, SNORM8
  //  coneOctY    | 8    | octant coordinate for cone normal, SNORM8
  //  coneAngle   | 8    | -sin(cone.angle),  SNORM8
  //  vertexPack  | 8    | vertex indices per 32 bits (1 or 2)
  //  ------------|:----:|----------------------------------------------
  //   Field.W    |      |
  //  ------------|:----:|----------------------------------------------
  //  packOffset  | 32   | index buffer value of the first vertex

  //
  // Note : the bitfield is not expanded in the struct due to differences in how
  //        GPU & CPU compilers pack bit-fields and endian-ness.

  union
  {
#if !defined(NDEBUG) && defined(_MSC_VER)
    struct
    {
      // warning, not portable
      unsigned bboxMinX : 8;
      unsigned bboxMinY : 8;
      unsigned bboxMinZ : 8;
      unsigned vertexMax : 8;

      unsigned bboxMaxX : 8;
      unsigned bboxMaxY : 8;
      unsigned bboxMaxZ : 8;
      unsigned primMax : 8;

      signed   coneOctX : 8;
      signed   coneOctY : 8;
      signed   coneAngle : 8;
      unsigned vertexPack : 8;

      unsigned packOffset : 32;
    } _debug;
#endif
    struct
    {
      uint32_t fieldX;
      uint32_t fieldY;
      uint32_t fieldZ;
      uint32_t fieldW;
    };
  };

  uint32_t getNumVertices() const { return unpack(fieldX, 8, 24) + 1; }
  void     setNumVertices(uint32_t num)
  {
    assert(num <= MAX_VERTEX_COUNT_LIMIT);
    fieldX |= pack(num - 1, 8, 24);
  }

  uint32_t getNumPrims() const { return unpack(fieldY, 8, 24) + 1; }
  void     setNumPrims(uint32_t num)
  {
    assert(num <= MAX_PRIMITIVE_COUNT_LIMIT);
    fieldY |= pack(num - 1, 8, 24);
  }

  uint32_t getNumVertexPack() const { return unpack(fieldZ, 8, 24); }
  void     setNumVertexPack(uint32_t num) { fieldZ |= pack(num, 8, 24); }

  uint32_t getPackOffset() const { return fieldW; }
  void     setPackOffset(uint32_t index) { fieldW = index; }

  uint32_t getVertexStart() const { return 0; }
  uint32_t getVertexSize() const
  {
    uint32_t vertexDiv   = getNumVertexPack();
    uint32_t vertexElems = ((getNumVertices() + vertexDiv - 1) / vertexDiv);

    return vertexElems;
  }

  uint32_t getPrimStart() const { return (getVertexStart() + getVertexSize() + 1) & (~1u); }
  uint32_t getPrimSize() const
  {
    uint32_t primDiv   = 4;
    uint32_t primElems = ((getNumPrims() * 3 + PACKBASIC_PRIMITIVE_INDICES_PER_FETCH - 1) / primDiv);

    return primElems;
  }

  // positions are relative to object's bbox treated as UNORM
  void setBBox(uint8_t const bboxMin[3], uint8_t const bboxMax[3])
  {
    fieldX |= pack(bboxMin[0], 8, 0) | pack(bboxMin[1], 8, 8) | pack(bboxMin[2], 8, 16);
    fieldY |= pack(bboxMax[0], 8, 0) | pack(bboxMax[1], 8, 8) | pack(bboxMax[2], 8, 16);
  }

  void getBBox(uint8_t bboxMin[3], uint8_t bboxMax[3]) const
  {
    bboxMin[0] = unpack(fieldX, 8, 0);
    bboxMin[0] = unpack(fieldX, 8, 8);
    bboxMin[0] = unpack(fieldX, 8, 16);

    bboxMax[0] = unpack(fieldY, 8, 0);
    bboxMax[0] = unpack(fieldY, 8, 8);
    bboxMax[0] = unpack(fieldY, 8, 16);
  }

  // uses octant encoding for cone Normal
  // positive angle means the cluster cannot be backface-culled
  // numbers are treated as SNORM
  void setCone(int8_t coneOctX, int8_t coneOctY, int8_t minusSinAngle)
  {
    uint8_t anglebits = minusSinAngle;
    fieldZ |= pack(coneOctX, 8, 0);
    fieldZ |= pack(coneOctY, 8, 8);
    fieldZ |= pack(minusSinAngle, 8, 16);
  }

  void getCone(int8_t& coneOctX, int8_t& coneOctY, int8_t& minusSinAngle) const
  {
    coneOctX      = unpack(fieldZ, 8, 0);
    coneOctY      = unpack(fieldZ, 8, 8);
    minusSinAngle = unpack(fieldZ, 8, 16);
  }

  MeshletPackBasicDesc()
  {
    fieldX = 0;
    fieldY = 0;
    fieldZ = 0;
    fieldW = 0;
  }
};

struct MeshletPackBasic
{

  // variable size
  //
  // aligned to PACKBASIC_ALIGN bytes
  // - first squence is either 16 or 32 bit indices per vertex
  //   (vertexPack is 2 or 1) respectively
  // - second sequence aligned to 8 bytes, primitive many 8 bit values
  //   
  //
  // { u32[numVertices/vertexPack ...], padding..., u8[(numPrimitives) * 3 ...] }

  union
  {
    uint32_t data32[1];
    uint16_t data16[1];
    uint8_t  data8[1];
  };

  inline void setVertexIndex(uint32_t PACKED_SIZE, uint32_t vertex, uint32_t vertexPack, uint32_t indexValue)
  {
#if 1
    if (vertexPack == 1){
      data32[vertex] = indexValue;
    }
    else {
      data16[vertex] = indexValue;
    }
#else
    uint32_t idx   = vertex / vertexPack;
    uint32_t shift = vertex % vertexPack;

    assert(idx < PACKED_SIZE);

    data32[idx] |= indexValue << (shift * 16);
#endif
  }

  inline uint32_t getVertexIndex(uint32_t vertex, uint32_t vertexPack) const
  {
#if 1
    return (vertexPack == 1) ? data32[vertex] : data16[vertex];
#else
    uint32_t idx   = vertex / vertexPack;
    uint32_t shift = vertex & (vertexPack-1);
    uint32_t bits  = vertexPack == 2 ? 16 : 0;

    uint32_t indexValue = data32[idx];
    indexValue <<= ((1 - shift) * bits);
    indexValue >>= (bits);
    return indexValue;
#endif
  }

  inline void setPrimIndices(uint32_t PACKED_SIZE, uint32_t prim, uint32_t primStart, const uint8_t indices[3])
  {
    uint32_t idx   = primStart * 4 + prim * 3;

    assert(idx < PACKED_SIZE * 4);

    data8[idx + 0] = indices[0];
    data8[idx + 1] = indices[1];
    data8[idx + 2] = indices[2];
  }

  inline void getPrimIndices(uint32_t prim, uint32_t primStart, uint8_t indices[3]) const
  {
    uint32_t idx = primStart * 4 + prim * 3;
    
    indices[0]   = data8[idx + 0];
    indices[1]   = data8[idx + 1];
    indices[2]   = data8[idx + 2];
  }
};

class PackBasicBuilder
{
public:
  //////////////////////////////////////////////////////////////////////////
  // Builder output
  // The provided builder functions operate on one triangle mesh at a time
  // and generate these outputs.

  struct MeshletGeometry
  {
    std::vector<PackBasicType>        meshletPacks;
    std::vector<MeshletPackBasicDesc> meshletDescriptors;
    std::vector<MeshletBbox>          meshletBboxes;
  };


  //////////////////////////////////////////////////////////////////////////
  // Builder configuration
private:
  // might want to template these instead of using MAX
  uint32_t m_maxVertexCount;
  uint32_t m_maxPrimitiveCount;
  bool     m_separateBboxes;

  // due to hw allocation granuarlity, good values are
  // vertex count = 32 or 64
  // primitive count = 40, 84 or 126
  //                   maximizes the fit into gl_PrimitiveIndices[128 * N - 4]
public:
  void setup(uint32_t maxVertexCount, uint32_t maxPrimitiveCount, bool separateBboxes = false)
  {
    assert(maxPrimitiveCount <= MAX_PRIMITIVE_COUNT_LIMIT);
    assert(maxVertexCount <= MAX_VERTEX_COUNT_LIMIT);

    m_maxVertexCount    = maxVertexCount;
    m_maxPrimitiveCount = maxPrimitiveCount;
    m_separateBboxes    = separateBboxes;

    {
      uint32_t indices = maxPrimitiveCount * 3;
      // align to PRIMITIVE_INDICES_PER_FETCH
      uint32_t indicesFit = (indices / PACKBASIC_PRIMITIVE_INDICES_PER_FETCH) * PACKBASIC_PRIMITIVE_INDICES_PER_FETCH;
      uint32_t numTrisFit = indicesFit / 3;

      assert(numTrisFit > 0);
      m_maxPrimitiveCount = numTrisFit;
    }
  }

  //////////////////////////////////////////////////////////////////////////
  // generate meshlets
private:
  void addMeshlet(MeshletGeometry& geometry, const PrimitiveCache& cache) const
  {
    uint32_t packOffset = uint32_t(geometry.meshletPacks.size());
    uint32_t vertexPack = cache.numVertexAllBits <= 16 ? 2 : 1;

    MeshletPackBasicDesc meshlet;
    meshlet.setNumPrims(cache.numPrims);
    meshlet.setNumVertices(cache.numVertices);
    meshlet.setNumVertexPack(vertexPack);
    meshlet.setPackOffset(packOffset);

    uint32_t vertexStart = meshlet.getVertexStart();
    uint32_t vertexSize  = meshlet.getVertexSize();

    uint32_t primStart = meshlet.getPrimStart();
    uint32_t primSize  = meshlet.getPrimSize();

    uint32_t packedSize = std::max(vertexStart + vertexSize, primStart + primSize);
    packedSize          = alignedSize(packedSize, PACKBASIC_ALIGN);

    geometry.meshletPacks.resize(geometry.meshletPacks.size() + packedSize, 0);
    geometry.meshletDescriptors.push_back(meshlet);

    MeshletPackBasic* pack = (MeshletPackBasic*)&geometry.meshletPacks[packOffset];

    {
      for(uint32_t v = 0; v < cache.numVertices; v++)
      {
        pack->setVertexIndex(packedSize, v, vertexPack, cache.vertices[v]);
      }

      uint32_t primStart = meshlet.getPrimStart();

      for(uint32_t p = 0; p < cache.numPrims; p++)
      {
        pack->setPrimIndices(packedSize, p, primStart, cache.primitives[p]);
      }
    }
  }

public:
  // Returns the number of successfully processed indices.
  // If the returned number is lower than provided input, use the number
  // as starting offset and create a new geometry description.
  template <class VertexIndexType>
  uint32_t buildMeshlets(MeshletGeometry& geometry, const uint32_t numIndices, const VertexIndexType* NV_RESTRICT indices) const
  {
    assert(m_maxPrimitiveCount <= MAX_PRIMITIVE_COUNT_LIMIT);
    assert(m_maxVertexCount <= MAX_VERTEX_COUNT_LIMIT);

    PrimitiveCache cache;
    cache.maxPrimitiveSize = m_maxPrimitiveCount;
    cache.maxVertexSize    = m_maxVertexCount;
    cache.reset();

    for(uint32_t i = 0; i < numIndices / 3; i++)
    {
      if(cache.cannotInsertBlock(indices[i * 3 + 0], indices[i * 3 + 1], indices[i * 3 + 2]))
      {
        // finish old and reset
        addMeshlet(geometry, cache);
        cache.reset();
      }
      cache.insert(indices[i * 3 + 0], indices[i * 3 + 1], indices[i * 3 + 2]);
    }
    if(!cache.empty())
    {
      addMeshlet(geometry, cache);
    }

    return numIndices;
  }


  void padTaskMeshlets(MeshletGeometry& geometry) const
  {
    if(geometry.meshletDescriptors.empty())
      return;

    // ensure we never have out-of-bounds memory access to array within task shader
    for(uint32_t i = 0; i < MESHLETS_PER_TASK - 1; i++)
    {
      geometry.meshletDescriptors.push_back(MeshletPackBasicDesc());
    }
  }

  //////////////////////////////////////////////////////////////////////////
  // generate early culling per meshlet

public:
  // bbox and cone angle
  void buildMeshletEarlyCulling(MeshletGeometry& geometry,
                                const float      objectBboxMin[3],
                                const float      objectBboxMax[3],
                                const float* NV_RESTRICT positions,
                                const size_t             positionStride) const
  {
    assert((positionStride % sizeof(float)) == 0);

    size_t positionMul = positionStride / sizeof(float);

    vec objectBboxExtent = vec(objectBboxMax) - vec(objectBboxMin);

    if(m_separateBboxes)
    {
      geometry.meshletBboxes.resize(geometry.meshletDescriptors.size());
    }

    for(size_t i = 0; i < geometry.meshletDescriptors.size(); i++)
    {
      MeshletPackBasicDesc&   meshlet = geometry.meshletDescriptors[i];
      const MeshletPackBasic* pack    = (const MeshletPackBasic*)&geometry.meshletPacks[meshlet.getPackOffset()];

      uint32_t primCount   = meshlet.getNumPrims();
      uint32_t primStart   = meshlet.getPrimStart();
      uint32_t vertexCount = meshlet.getNumVertices();
      uint32_t vertexPack  = meshlet.getNumVertexPack();

      vec bboxMin = vec(FLT_MAX);
      vec bboxMax = vec(-FLT_MAX);

      vec avgNormal = vec(0.0f);
      vec triNormals[MAX_PRIMITIVE_COUNT_LIMIT];

      // skip unset
      if(vertexCount == 1)
        continue;

      for(uint32_t p = 0; p < primCount; p++)
      {
        uint8_t  indices[3];
        uint32_t idxA;
        uint32_t idxB;
        uint32_t idxC;

        pack->getPrimIndices(p, primStart, indices);
        idxA = pack->getVertexIndex(indices[0], vertexPack);
        idxB = pack->getVertexIndex(indices[1], vertexPack);
        idxC = pack->getVertexIndex(indices[2], vertexPack);

        vec posA = vec(&positions[idxA * positionMul]);
        vec posB = vec(&positions[idxB * positionMul]);
        vec posC = vec(&positions[idxC * positionMul]);

        {
          // bbox
          bboxMin = vec_min(bboxMin, posA);
          bboxMin = vec_min(bboxMin, posB);
          bboxMin = vec_min(bboxMin, posC);

          bboxMax = vec_max(bboxMax, posA);
          bboxMax = vec_max(bboxMax, posB);
          bboxMax = vec_max(bboxMax, posC);
        }

        {
          // cone
          vec   cross  = vec_cross(posB - posA, posC - posA);
          float length = vec_length(cross);

          vec normal;
          if(length > FLT_EPSILON)
          {
            normal = cross * (1.0f / length);
          }
          else
          {
            normal = cross;
          }

          avgNormal     = avgNormal + normal;
          triNormals[p] = normal;
        }
      }

      if(m_separateBboxes)
      {
        geometry.meshletBboxes[i].bboxMin[0] = bboxMin.x;
        geometry.meshletBboxes[i].bboxMin[1] = bboxMin.y;
        geometry.meshletBboxes[i].bboxMin[2] = bboxMin.z;
        geometry.meshletBboxes[i].bboxMax[0] = bboxMax.x;
        geometry.meshletBboxes[i].bboxMax[1] = bboxMax.y;
        geometry.meshletBboxes[i].bboxMax[2] = bboxMax.z;
      }

      {
        // bbox
        // truncate min relative to object min
        bboxMin = bboxMin - vec(objectBboxMin);
        bboxMax = bboxMax - vec(objectBboxMin);
        bboxMin = bboxMin / objectBboxExtent;
        bboxMax = bboxMax / objectBboxExtent;

        // snap to grid
        const int gridBits = 8;
        const int gridLast = (1 << gridBits) - 1;
        uint8_t   gridMin[3];
        uint8_t   gridMax[3];

        gridMin[0] = std::max(0, std::min(int(truncf(bboxMin.x * float(gridLast))), gridLast - 1));
        gridMin[1] = std::max(0, std::min(int(truncf(bboxMin.y * float(gridLast))), gridLast - 1));
        gridMin[2] = std::max(0, std::min(int(truncf(bboxMin.z * float(gridLast))), gridLast - 1));
        gridMax[0] = std::max(0, std::min(int(ceilf(bboxMax.x * float(gridLast))), gridLast));
        gridMax[1] = std::max(0, std::min(int(ceilf(bboxMax.y * float(gridLast))), gridLast));
        gridMax[2] = std::max(0, std::min(int(ceilf(bboxMax.z * float(gridLast))), gridLast));

        meshlet.setBBox(gridMin, gridMax);
      }

      {
        // potential improvement, instead of average maybe use
        // http://www.cs.technion.ac.il/~cggc/files/gallery-pdfs/Barequet-1.pdf

        float len = vec_length(avgNormal);
        if(len > FLT_EPSILON)
        {
          avgNormal = avgNormal / len;
        }
        else
        {
          avgNormal = vec(0.0f);
        }

        vec    packed = float32x3_to_octn_precise(avgNormal, 16);
        int8_t coneX  = std::min(127, std::max(-127, int32_t(packed.x * 127.0f)));
        int8_t coneY  = std::min(127, std::max(-127, int32_t(packed.y * 127.0f)));

        // post quantization normal
        avgNormal = oct_to_float32x3(vec(float(coneX) / 127.0f, float(coneY) / 127.0f, 0.0f));

        float mindot = 1.0f;
        for(unsigned int p = 0; p < primCount; p++)
        {
          mindot = std::min(mindot, vec_dot(triNormals[p], avgNormal));
        }

        // apply safety delta due to quantization
        mindot -= 1.0f / 127.0f;
        mindot = std::max(-1.0f, mindot);

        // positive value for cluster not being backface cullable (normals > 90°)
        int8_t coneAngle = 127;
        if(mindot > 0)
        {
          // otherwise store -sin(cone angle)
          // we test against dot product (cosine) so this is equivalent to cos(cone angle + 90°)
          float angle = -sinf(acosf(mindot));
          coneAngle   = std::max(-127, std::min(127, int32_t(angle * 127.0f)));
        }

        meshlet.setCone(coneX, coneY, coneAngle);
      }
    }
  }

  //////////////////////////////////////////////////////////////////////////

  template <class VertexIndexType>
  StatusCode errorCheck(const MeshletGeometry& geometry,
                        uint32_t               minVertex,
                        uint32_t               maxVertex,
                        uint32_t               numIndices,
                        const VertexIndexType* NV_RESTRICT indices) const
  {
    uint32_t compareTris = 0;

    for(size_t i = 0; i < geometry.meshletDescriptors.size(); i++)
    {
      const MeshletPackBasicDesc& meshlet = geometry.meshletDescriptors[i];
      const MeshletPackBasic*     pack    = (const MeshletPackBasic*)&geometry.meshletPacks[meshlet.getPackOffset()];

      uint32_t primCount   = meshlet.getNumPrims();
      uint32_t primStart   = meshlet.getPrimStart();
      uint32_t vertexCount = meshlet.getNumVertices();
      uint32_t vertexPack  = meshlet.getNumVertexPack();

      // skip unset
      if(vertexCount == 1)
        continue;

      for(uint32_t p = 0; p < primCount; p++)
      {
        uint8_t blockIndices[3];
        pack->getPrimIndices(p, primStart, blockIndices);

        if(blockIndices[0] >= m_maxVertexCount || blockIndices[1] >= m_maxVertexCount || blockIndices[2] >= m_maxVertexCount)
        {
          return STATUS_PRIM_OUT_OF_BOUNDS;
        }

        uint32_t idxA = pack->getVertexIndex(blockIndices[0], vertexPack);
        uint32_t idxB = pack->getVertexIndex(blockIndices[1], vertexPack);
        uint32_t idxC = pack->getVertexIndex(blockIndices[2], vertexPack);

        if(idxA < minVertex || idxA > maxVertex || idxB < minVertex || idxB > maxVertex || idxC < minVertex || idxC > maxVertex)
        {
          return STATUS_VERTEX_OUT_OF_BOUNDS;
        }

        uint32_t refA = 0;
        uint32_t refB = 0;
        uint32_t refC = 0;

        while(refA == refB || refA == refC || refB == refC)
        {
          if(compareTris * 3 + 2 >= numIndices)
          {
            return STATUS_MISMATCH_INDICES;
          }
          refA = indices[compareTris * 3 + 0];
          refB = indices[compareTris * 3 + 1];
          refC = indices[compareTris * 3 + 2];
          compareTris++;
        }

        if(refA != idxA || refB != idxB || refC != idxC)
        {
          return STATUS_MISMATCH_INDICES;
        }
      }
    }

    return STATUS_NO_ERROR;
  }

  void appendStats(const MeshletGeometry& geometry, Stats& stats) const
  {
    if(geometry.meshletDescriptors.empty())
    {
      return;
    }

    stats.meshletsStored += geometry.meshletDescriptors.size();

    double primloadAvg   = 0;
    double primloadVar   = 0;
    double vertexloadAvg = 0;
    double vertexloadVar = 0;

    size_t meshletsTotal = 0;
    for(size_t i = 0; i < geometry.meshletDescriptors.size(); i++)
    {
      const MeshletPackBasicDesc& meshlet     = geometry.meshletDescriptors[i];
      uint32_t                    primCount   = meshlet.getNumPrims();
      uint32_t                    vertexCount = meshlet.getNumVertices();

      if(vertexCount == 1)
      {
        continue;
      }

      meshletsTotal++;

      stats.vertexTotal += vertexCount;
      stats.primTotal += primCount;
      primloadAvg += double(primCount) / double(m_maxPrimitiveCount);
      vertexloadAvg += double(vertexCount) / double(m_maxVertexCount);

      int8_t coneX;
      int8_t coneY;
      int8_t coneAngle;
      meshlet.getCone(coneX, coneY, coneAngle);
      stats.backfaceTotal += coneAngle < 0 ? 1 : 0;
    }

    stats.meshletsTotal += meshletsTotal;

    double statsNum = meshletsTotal ? double(meshletsTotal) : 1.0;

    primloadAvg /= statsNum;
    vertexloadAvg /= statsNum;
    for(size_t i = 0; i < geometry.meshletDescriptors.size(); i++)
    {
      const MeshletPackBasicDesc& meshlet     = geometry.meshletDescriptors[i];
      uint32_t                    primCount   = meshlet.getNumPrims();
      uint32_t                    vertexCount = meshlet.getNumVertices();
      double                      diff;

      diff = primloadAvg - ((double(primCount) / double(m_maxPrimitiveCount)));
      primloadVar += diff * diff;

      diff = vertexloadAvg - ((double(vertexCount) / double(m_maxVertexCount)));
      vertexloadVar += diff * diff;
    }
    primloadVar /= statsNum;
    vertexloadVar /= statsNum;

    stats.primloadAvg += primloadAvg;
    stats.primloadVar += primloadVar;
    stats.vertexloadAvg += vertexloadAvg;
    stats.vertexloadVar += vertexloadVar;
    stats.appended += 1.0;
  }
};
}  // namespace NVMeshlet

#endif
