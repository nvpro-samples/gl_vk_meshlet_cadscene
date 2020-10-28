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

#ifndef _NV_MESHLET_ARRAY_H__
#define _NV_MESHLET_ARRAY_H__

#include "nvmeshlet_builder.hpp"

namespace NVMeshlet {

// The default shown here packs uint8 tightly, and makes them accessible as 64-bit load.
// Keep in sync with shader configuration!

// how many indices are fetched per thread, 8 or 4
static const uint32_t ARRAY_PRIMITIVE_INDICES_PER_FETCH = 8;

// Higher values mean slightly more wasted memory, but allow to use greater offsets within
// the few bits we have, resulting in a higher total amount of triangles and vertices.
static const uint32_t ARRAY_PRIMITIVE_PACKING_ALIGNMENT = 32;  // must be multiple of PRIMITIVE_INDICES_PER_FETCH
static const uint32_t ARRAY_VERTEX_PACKING_ALIGNMENT    = 16;

inline uint64_t arrayIndicesAlignedSize(uint64_t size)
{
  // To be able to store different data of the meshlet (prim & vertex indices) in the same buffer,
  // we need to have a common alignment that keeps all the data natural aligned.

  static const uint64_t align =
      std::max(sizeof(uint8_t) * ARRAY_PRIMITIVE_PACKING_ALIGNMENT, sizeof(uint32_t) * ARRAY_VERTEX_PACKING_ALIGNMENT);
  static_assert(align % sizeof(uint8_t) * ARRAY_PRIMITIVE_PACKING_ALIGNMENT == 0, "nvmeshlet failed common align");
  static_assert(align % sizeof(uint32_t) * ARRAY_VERTEX_PACKING_ALIGNMENT == 0, "nvmeshlet failed common align");

  return ((size + align - 1) / align) * align;
}

struct MeshletArrayDesc
{
  // A Meshlet contains a set of unique vertices
  // and a group of primitives that are defined by
  // indices into this local set of vertices.
  //
  // The information here is used by a single
  // mesh shader's workgroup to execute vertex
  // and primitive shading.
  // It is packed into single "uvec4"/"uint4" value
  // so the hardware can leverage 128-bit loads in the
  // shading languages.
  // The offsets used here are for the appropriate
  // indices arrays.
  //
  // A bounding box as well as an angled cone is stored to allow
  // quick culling in the task shader.
  // The current packing is just a basic implementation, that
  // may be customized, but ideally fits within 128 bit.

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
  //  vertexBegin | 20   | offset to the first vertex index, times alignment
  //  coneOctX    | 8    | octant coordinate for cone normal, SNORM8
  //  coneAngleLo | 4    | lower 4 bits of -sin(cone.angle),  SNORM8
  //  ------------|:----:|----------------------------------------------
  //   Field.W    |      |
  //  ------------|:----:|----------------------------------------------
  //  primBegin   | 20   | offset to the first primitive index, times alignment
  //  coneOctY    | 8    | octant coordinate for cone normal, SNORM8
  //  coneAngleHi | 4    | higher 4 bits of -sin(cone.angle), SNORM8
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

      unsigned vertexBegin : 20;
      signed   coneOctX : 8;
      unsigned coneAngleLo : 4;

      unsigned primBegin : 20;
      signed   coneOctY : 8;
      unsigned coneAngleHi : 4;
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

  uint32_t getVertexBegin() const { return unpack(fieldZ, 20, 0) * ARRAY_VERTEX_PACKING_ALIGNMENT; }
  void     setVertexBegin(uint32_t begin)
  {
    assert(begin % ARRAY_VERTEX_PACKING_ALIGNMENT == 0);
    assert(begin / ARRAY_VERTEX_PACKING_ALIGNMENT < ((1 << 20) - 1));
    fieldZ |= pack(begin / ARRAY_VERTEX_PACKING_ALIGNMENT, 20, 0);
  }

  uint32_t getPrimBegin() const { return unpack(fieldW, 20, 0) * ARRAY_PRIMITIVE_PACKING_ALIGNMENT; }
  void     setPrimBegin(uint32_t begin)
  {
    assert(begin % ARRAY_PRIMITIVE_PACKING_ALIGNMENT == 0);
    assert(begin / ARRAY_PRIMITIVE_PACKING_ALIGNMENT < ((1 << 20) - 1));
    fieldW |= pack(begin / ARRAY_PRIMITIVE_PACKING_ALIGNMENT, 20, 0);
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
    bboxMin[1] = unpack(fieldX, 8, 8);
    bboxMin[2] = unpack(fieldX, 8, 16);

    bboxMax[0] = unpack(fieldY, 8, 0);
    bboxMax[1] = unpack(fieldY, 8, 8);
    bboxMax[2] = unpack(fieldY, 8, 16);
  }

  // uses octant encoding for cone Normal
  // positive angle means the cluster cannot be backface-culled
  // numbers are treated as SNORM
  void setCone(int8_t coneOctX, int8_t coneOctY, int8_t minusSinAngle)
  {
    uint8_t anglebits = minusSinAngle;
    fieldZ |= pack(coneOctX, 8, 20);
    fieldW |= pack(coneOctY, 8, 20);
    fieldZ |= pack((anglebits >> 0) & 0xF, 4, 28);
    fieldW |= pack((anglebits >> 4) & 0xF, 4, 28);
  }

  void getCone(int8_t& coneOctX, int8_t& coneOctY, int8_t& minusSinAngle) const
  {
    coneOctX      = unpack(fieldZ, 8, 20);
    coneOctY      = unpack(fieldW, 8, 20);
    minusSinAngle = unpack(fieldZ, 4, 28) | (unpack(fieldW, 4, 28) << 4);
  }

  MeshletArrayDesc()
  {
    fieldX = 0;
    fieldY = 0;
    fieldZ = 0;
    fieldW = 0;
  }

  static bool isPrimBeginLegal(uint32_t begin) { return begin / ARRAY_PRIMITIVE_PACKING_ALIGNMENT < ((1 << 20) - 1); }

  static bool isVertexBeginLegal(uint32_t begin) { return begin / ARRAY_VERTEX_PACKING_ALIGNMENT < ((1 << 20) - 1); }
};


//////////////////////////////////////////////////////////////////////////

template <class VertexIndexType>
class ArrayBuilder
{
public:
  //////////////////////////////////////////////////////////////////////////
  // Builder output
  // The provided builder functions operate on one triangle mesh at a time
  // and generate these outputs.

  struct MeshletGeometry
  {
    // The vertex indices are similar to provided to the provided
    // triangle index buffer. Instead of each triangle using 3 vertex indices,
    // each meshlet holds a unique set of variable vertex indices.
    std::vector<VertexIndexType> vertexIndices;

    // Each triangle is using 3 primitive indices, these indices
    // are local to the meshlet's unique set of vertices.
    // Due to alignment the number of primitiveIndices != input triangle indices.
    std::vector<PrimitiveIndexType> primitiveIndices;

    // Each meshlet contains offsets into the above arrays.
    std::vector<MeshletArrayDesc> meshletDescriptors;
    std::vector<MeshletBbox>      meshletBboxes;
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

    m_separateBboxes = separateBboxes;

    m_maxVertexCount = maxVertexCount;
    // we may reduce the number of actual triangles a bit to simplify
    // index loader logic in shader. By using less primitives we
    // guarantee to not overshoot the gl_PrimitiveIndices array when using the 32-bit
    // write intrinsic.

    {
      uint32_t indices = maxPrimitiveCount * 3;
      // align to PRIMITIVE_INDICES_PER_FETCH
      uint32_t indicesFit = (indices / ARRAY_PRIMITIVE_INDICES_PER_FETCH) * ARRAY_PRIMITIVE_INDICES_PER_FETCH;
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
    MeshletArrayDesc meshlet;
    meshlet.setNumPrims(cache.numPrims);
    meshlet.setNumVertices(cache.numVertices);
    meshlet.setPrimBegin(uint32_t(geometry.primitiveIndices.size()));
    meshlet.setVertexBegin(uint32_t(geometry.vertexIndices.size()));

    for(uint32_t v = 0; v < cache.numVertices; v++)
    {
      geometry.vertexIndices.push_back(cache.vertices[v]);
    }

    // pad with existing values to aid compression

    for(uint32_t p = 0; p < cache.numPrims; p++)
    {
      geometry.primitiveIndices.push_back(cache.primitives[p][0]);
      geometry.primitiveIndices.push_back(cache.primitives[p][1]);
      geometry.primitiveIndices.push_back(cache.primitives[p][2]);
    }

    while((geometry.vertexIndices.size() % ARRAY_VERTEX_PACKING_ALIGNMENT) != 0)
    {
      geometry.vertexIndices.push_back(cache.vertices[cache.numVertices - 1]);
    }
    size_t idx = 0;
    while((geometry.primitiveIndices.size() % ARRAY_PRIMITIVE_PACKING_ALIGNMENT) != 0)
    {
      geometry.primitiveIndices.push_back(cache.primitives[cache.numPrims - 1][idx % 3]);
      idx++;
    }

    geometry.meshletDescriptors.push_back(meshlet);
  }

public:
  // Returns the number of successfully processed indices.
  // If the returned number is lower than provided input, use the number
  // as starting offset and create a new geometry description.
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
      if(cache.cannotInsert(indices[i * 3 + 0], indices[i * 3 + 1], indices[i * 3 + 2]))
      {
        // finish old and reset
        addMeshlet(geometry, cache);
        cache.reset();

        // if we exhausted the index buffers, return early
        if(!MeshletArrayDesc::isPrimBeginLegal(uint32_t(geometry.primitiveIndices.size()))
           || !MeshletArrayDesc::isVertexBeginLegal(uint32_t(geometry.vertexIndices.size())))
        {
          return i * 3;
        }
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
      geometry.meshletDescriptors.push_back(MeshletArrayDesc());
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

    if (m_separateBboxes){
      geometry.meshletBboxes.resize(geometry.meshletDescriptors.size());
    }

    for(size_t i = 0; i < geometry.meshletDescriptors.size(); i++)
    {
      MeshletArrayDesc& meshlet = geometry.meshletDescriptors[i];

      uint32_t primCount   = meshlet.getNumPrims();
      uint32_t vertexCount = meshlet.getNumVertices();

      uint32_t primBegin   = meshlet.getPrimBegin();
      uint32_t vertexBegin = meshlet.getVertexBegin();

      vec bboxMin = vec(FLT_MAX);
      vec bboxMax = vec(-FLT_MAX);

      vec avgNormal = vec(0.0f);
      vec triNormals[MAX_PRIMITIVE_COUNT_LIMIT];

      // skip unset
      if(vertexCount == 1)
        continue;

      for(uint32_t p = 0; p < primCount; p++)
      {
        const uint32_t primStride = 3;

        uint32_t idxA = geometry.primitiveIndices[primBegin + p * primStride + 0];
        uint32_t idxB = geometry.primitiveIndices[primBegin + p * primStride + 1];
        uint32_t idxC = geometry.primitiveIndices[primBegin + p * primStride + 2];

        idxA = geometry.vertexIndices[vertexBegin + idxA];
        idxB = geometry.vertexIndices[vertexBegin + idxB];
        idxC = geometry.vertexIndices[vertexBegin + idxC];

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

      if (m_separateBboxes)
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

  StatusCode errorCheck(const MeshletGeometry& geometry,
                        uint32_t               minVertex,
                        uint32_t               maxVertex,
                        uint32_t               numIndices,
                        const VertexIndexType* NV_RESTRICT indices) const
  {
    uint32_t compareTris = 0;

    for(size_t i = 0; i < geometry.meshletDescriptors.size(); i++)
    {
      const MeshletArrayDesc& meshlet = geometry.meshletDescriptors[i];

      uint32_t primCount   = meshlet.getNumPrims();
      uint32_t vertexCount = meshlet.getNumVertices();

      uint32_t primBegin   = meshlet.getPrimBegin();
      uint32_t vertexBegin = meshlet.getVertexBegin();

      // skip unset
      if(vertexCount == 1)
        continue;

      for(uint32_t p = 0; p < primCount; p++)
      {
        const uint32_t primStride = 3;

        uint32_t idxA = geometry.primitiveIndices[primBegin + p * primStride + 0];
        uint32_t idxB = geometry.primitiveIndices[primBegin + p * primStride + 1];
        uint32_t idxC = geometry.primitiveIndices[primBegin + p * primStride + 2];

        if(idxA >= m_maxVertexCount || idxB >= m_maxVertexCount || idxC >= m_maxVertexCount)
        {
          return STATUS_PRIM_OUT_OF_BOUNDS;
        }

        idxA = geometry.vertexIndices[vertexBegin + idxA];
        idxB = geometry.vertexIndices[vertexBegin + idxB];
        idxC = geometry.vertexIndices[vertexBegin + idxC];

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
    stats.primIndices += geometry.primitiveIndices.size();
    stats.vertexIndices += geometry.vertexIndices.size();

    double primloadAvg   = 0;
    double primloadVar   = 0;
    double vertexloadAvg = 0;
    double vertexloadVar = 0;

    size_t meshletsTotal = 0;
    for(size_t i = 0; i < geometry.meshletDescriptors.size(); i++)
    {
      const MeshletArrayDesc& meshlet     = geometry.meshletDescriptors[i];
      uint32_t                primCount   = meshlet.getNumPrims();
      uint32_t                vertexCount = meshlet.getNumVertices();

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
      const MeshletArrayDesc& meshlet     = geometry.meshletDescriptors[i];
      uint32_t                primCount   = meshlet.getNumPrims();
      uint32_t                vertexCount = meshlet.getNumVertices();
      double                  diff;

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
