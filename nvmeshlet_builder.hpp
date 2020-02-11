/* Copyright (c) 2014-2018, NVIDIA CORPORATION. All rights reserved.
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

/* feedback: Christoph Kubisch <ckubisch@nvidia.com> */

#ifndef _NV_MESHLET_BUILDER_H__
#define _NV_MESHLET_BUILDER_H__

#include <NvFoundation.h>
#include <algorithm>
#include <cstdint>
#if (defined(NV_X86) || defined(NV_X64)) && defined(_MSC_VER)
#include <intrin.h>
#endif
#include <vector>
#include <stdio.h>


namespace NVMeshlet {
// Each Meshlet can have a varying count of its maximum number
// of vertices and primitives. We hardcode a few absolute maxima
// to accellerate some of the functions and allow usage of
// smaller datastructures.

// The builder, however, is configurable to use smaller maxima,
// which is recommended.

// The limits below are hard limits due to the encoding chosen for the
// meshlet descriptor. Actual hw-limits can be higher, but typically
// do not make things faster due to large on-chip allocation.

static const int MAX_VERTEX_COUNT_LIMIT    = 256;
static const int MAX_PRIMITIVE_COUNT_LIMIT = 256;

// use getTaskPaddedElements
static const uint32_t MESHLETS_PER_TASK = 32;

// must not change
typedef uint8_t PrimitiveIndexType;  // must store [0,MAX_VERTEX_COUNT_LIMIT-1]

// We allow two different type of primitive index packings.
// The first is preferred, but yields slightly greater code complexity.
enum PrimitiveIndexPacking
{
  // Dense array of multiple uint8s, 3 uint8s per primitive.
  // Least waste, can partially use 32-bit storage intrinsic for writing to gl_PrimitiveIndices
  PRIMITIVE_PACKING_TIGHT_UINT8,

  // Same as above but we may use less triangles to simplify loader logic.
  // We guarantee that all indices can be safely written to the gl_PrimitiveIndices array
  // using the 32-bit write intrinsic in the shader.
  PRIMITIVE_PACKING_FITTED_UINT8,

  // 4 uint8s per primitive, indices in first three 8-bit
  // makes decoding an individual triangle easy, but sacrifices bandwidth/storage
  NVMESHLET_PACKING_TRIANGLE_UINT32,
};

// The default shown here packs uint8 tightly, and makes them accessible as 64-bit load.
// Keep in sync with shader configuration!

static const PrimitiveIndexPacking PRIMITIVE_PACKING = PRIMITIVE_PACKING_FITTED_UINT8;
// how many indices are fetched per thread, 8 or 4
static const uint32_t PRIMITIVE_INDICES_PER_FETCH = 8;

// Higher values mean slightly more wasted memory, but allow to use greater offsets within
// the few bits we have, resulting in a higher total amount of triangles and vertices.
static const uint32_t PRIMITIVE_PACKING_ALIGNMENT = 32;  // must be multiple of PRIMITIVE_BITS_PER_FETCH
static const uint32_t VERTEX_PACKING_ALIGNMENT    = 16;

inline uint32_t computeTasksCount(uint32_t numMeshlets)
{
  return (numMeshlets + MESHLETS_PER_TASK - 1) / MESHLETS_PER_TASK;
}

inline uint32_t computePackedPrimitiveCount(uint32_t numTris)
{
  if(PRIMITIVE_PACKING != PRIMITIVE_PACKING_FITTED_UINT8)
    return numTris;

  uint32_t indices = numTris * 3;
  // align to PRIMITIVE_INDICES_PER_FETCH
  uint32_t indicesFit = (indices / PRIMITIVE_INDICES_PER_FETCH) * PRIMITIVE_INDICES_PER_FETCH;
  uint32_t numTrisFit = indicesFit / 3;
  ;
  assert(numTrisFit > 0);
  return numTrisFit;
}


struct MeshletDesc
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

  uint32_t getVertexBegin() const { return unpack(fieldZ, 20, 0) * VERTEX_PACKING_ALIGNMENT; }
  void     setVertexBegin(uint32_t begin)
  {
    assert(begin % VERTEX_PACKING_ALIGNMENT == 0);
    assert(begin / VERTEX_PACKING_ALIGNMENT < ((1 << 20) - 1));
    fieldZ |= pack(begin / VERTEX_PACKING_ALIGNMENT, 20, 0);
  }

  uint32_t getPrimBegin() const { return unpack(fieldW, 20, 0) * PRIMITIVE_PACKING_ALIGNMENT; }
  void     setPrimBegin(uint32_t begin)
  {
    assert(begin % PRIMITIVE_PACKING_ALIGNMENT == 0);
    assert(begin / PRIMITIVE_PACKING_ALIGNMENT < ((1 << 20) - 1));
    fieldW |= pack(begin / PRIMITIVE_PACKING_ALIGNMENT, 20, 0);
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
    fieldZ |= pack(coneOctX, 8, 20) | pack((anglebits >> 0) & 0xF, 4, 28);
    fieldW |= pack(coneOctY, 8, 20) | pack((anglebits >> 4) & 0xF, 4, 28);
  }

  void getCone(int8_t& coneOctX, int8_t& coneOctY, int8_t& minusSinAngle) const
  {
    coneOctX      = unpack(fieldZ, 8, 20);
    coneOctY      = unpack(fieldW, 8, 20);
    minusSinAngle = unpack(fieldZ, 4, 28) | (unpack(fieldW, 4, 28) << 4);
  }

  MeshletDesc() { memset(this, 0, sizeof(MeshletDesc)); }

  static uint32_t pack(uint32_t value, int width, int offset)
  {
    return (uint32_t)((value & ((1 << width) - 1)) << offset);
  }
  static uint32_t unpack(uint32_t value, int width, int offset)
  {
    return (uint32_t)((value >> offset) & ((1 << width) - 1));
  }

  static bool isPrimBeginLegal(uint32_t begin) { return begin / PRIMITIVE_PACKING_ALIGNMENT < ((1 << 20) - 1); }

  static bool isVertexBeginLegal(uint32_t begin) { return begin / VERTEX_PACKING_ALIGNMENT < ((1 << 20) - 1); }
};

inline uint64_t computeCommonAlignedSize(uint64_t size)
{
  // To be able to store different data of the meshlet (desc, prim & vertex indices) in the same buffer,
  // we need to have a common alignment that keeps all the data natural aligned.

  static const uint64_t align = std::max(std::max(sizeof(MeshletDesc), sizeof(uint8_t) * PRIMITIVE_PACKING_ALIGNMENT),
                                         sizeof(uint32_t) * VERTEX_PACKING_ALIGNMENT);
  static_assert(align % sizeof(MeshletDesc) == 0, "nvmeshlet failed common align");
  static_assert(align % sizeof(uint8_t) * PRIMITIVE_PACKING_ALIGNMENT == 0, "nvmeshlet failed common align");
  static_assert(align % sizeof(uint32_t) * VERTEX_PACKING_ALIGNMENT == 0, "nvmeshlet failed common align");

  return ((size + align - 1) / align) * align;
}

inline uint64_t computeIndicesAlignedSize(uint64_t size)
{
  // To be able to store different data of the meshlet (prim & vertex indices) in the same buffer,
  // we need to have a common alignment that keeps all the data natural aligned.

  static const uint64_t align = std::max(sizeof(uint8_t) * PRIMITIVE_PACKING_ALIGNMENT, sizeof(uint32_t) * VERTEX_PACKING_ALIGNMENT);
  static_assert(align % sizeof(uint8_t) * PRIMITIVE_PACKING_ALIGNMENT == 0, "nvmeshlet failed common align");
  static_assert(align % sizeof(uint32_t) * VERTEX_PACKING_ALIGNMENT == 0, "nvmeshlet failed common align");

  return ((size + align - 1) / align) * align;
}

//////////////////////////////////////////////////////////////////////////
//

struct Stats
{
  size_t meshletsTotal = 0;
  // slightly more due to task-shader alignment
  size_t meshletsStored = 0;

  // number of meshlets that can be backface cluster culled at all
  // due to similar normals
  size_t backfaceTotal = 0;

  size_t primIndices = 0;
  size_t primTotal   = 0;

  size_t vertexIndices = 0;
  size_t vertexTotal   = 0;

  // used when we sum multiple stats into a single to
  // compute averages of the averages/variances below.

  size_t appended = 0;

  double primloadAvg   = 0.f;
  double primloadVar   = 0.f;
  double vertexloadAvg = 0.f;
  double vertexloadVar = 0.f;

  void append(const Stats& other)
  {
    meshletsTotal += other.meshletsTotal;
    meshletsStored += other.meshletsStored;
    backfaceTotal += other.backfaceTotal;

    primIndices += other.primIndices;
    vertexIndices += other.vertexIndices;
    vertexTotal += other.vertexTotal;
    primTotal += other.primTotal;

    appended += other.appended;
    primloadAvg += other.primloadAvg;
    primloadVar += other.primloadVar;
    vertexloadAvg += other.vertexloadAvg;
    vertexloadVar += other.vertexloadVar;
  }

  void fprint(FILE* log) const
  {
    if(!appended || !meshletsTotal)
      return;

    double fprimloadAvg   = primloadAvg / double(appended);
    double fprimloadVar   = primloadVar / double(appended);
    double fvertexloadAvg = vertexloadAvg / double(appended);
    double fvertexloadVar = vertexloadVar / double(appended);

    double statsNum    = double(meshletsTotal);
    double backfaceAvg = double(backfaceTotal) / statsNum;

    double primWaste    = double(primIndices) / double(primTotal * 3) - 1.0;
    double vertexWaste  = double(vertexIndices) / double(vertexTotal) - 1.0;
    double meshletWaste = double(meshletsStored) / double(meshletsTotal) - 1.0;

    fprintf(log,
            "meshlets; %7zd; prim; %9zd; %.2f; vertex; %9zd; %.2f; backface; %.2f; waste; v; %.2f; p; %.2f; m; %.2f\n", meshletsTotal,
            primTotal, fprimloadAvg, vertexTotal, fvertexloadAvg, backfaceAvg, vertexWaste, primWaste, meshletWaste);
  }
};

//////////////////////////////////////////////////////////////////////////
// simple vector class to reduce dependencies

struct vec
{
  float x;
  float y;
  float z;

  vec() {}
  vec(float v)
      : x(v)
      , y(v)
      , z(v)
  {
  }
  vec(float _x, float _y, float _z)
      : x(_x)
      , y(_y)
      , z(_z)
  {
  }
  vec(const float* v)
      : x(v[0])
      , y(v[1])
      , z(v[2])
  {
  }
};

inline vec vec_min(const vec& a, const vec& b)
{
  return vec(std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z));
}
inline vec vec_max(const vec& a, const vec& b)
{
  return vec(std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z));
}
inline vec operator+(const vec& a, const vec& b)
{
  return vec(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline vec operator-(const vec& a, const vec& b)
{
  return vec(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline vec operator/(const vec& a, const vec& b)
{
  return vec(a.x / b.x, a.y / b.y, a.z / b.z);
}
inline vec operator*(const vec& a, const vec& b)
{
  return vec(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline vec operator*(const vec& a, const float b)
{
  return vec(a.x * b, a.y * b, a.z * b);
}
inline vec vec_floor(const vec& a)
{
  return vec(floorf(a.x), floorf(a.y), floorf(a.z));
}
inline vec vec_clamp(const vec& a, const float lowerV, const float upperV)
{
  return vec(std::max(std::min(upperV, a.x), lowerV), std::max(std::min(upperV, a.y), lowerV),
             std::max(std::min(upperV, a.z), lowerV));
}
inline vec vec_cross(const vec& a, const vec& b)
{
  return vec(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
inline float vec_dot(const vec& a, const vec& b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline float vec_length(const vec& a)
{
  return sqrtf(vec_dot(a, a));
}
inline vec vec_normalize(const vec& a)
{
  float len = vec_length(a);
  return a * 1.0f / len;
}

// all oct functions derived from "A Survey of Efficient Representations for Independent Unit Vectors"
// http://jcgt.org/published/0003/02/01/paper.pdf
// Returns +/- 1
inline vec oct_signNotZero(vec v)
{
  // leaves z as is
  return vec((v.x >= 0.0f) ? +1.0f : -1.0f, (v.y >= 0.0f) ? +1.0f : -1.0f, 1.0f);
}

// Assume normalized input. Output is on [-1, 1] for each component.
inline vec float32x3_to_oct(vec v)
{
  // Project the sphere onto the octahedron, and then onto the xy plane
  vec p = vec(v.x, v.y, 0) * (1.0f / (fabsf(v.x) + fabsf(v.y) + fabsf(v.z)));
  // Reflect the folds of the lower hemisphere over the diagonals
  return (v.z <= 0.0f) ? vec(1.0f - fabsf(p.y), 1.0f - fabsf(p.x), 0.0f) * oct_signNotZero(p) : p;
}

inline vec oct_to_float32x3(vec e)
{
  vec v = vec(e.x, e.y, 1.0f - fabsf(e.x) - fabsf(e.y));
  if(v.z < 0.0f)
  {
    v = vec(1.0f - fabs(v.y), 1.0f - fabs(v.x), v.z) * oct_signNotZero(v);
  }
  return vec_normalize(v);
}

inline vec float32x3_to_octn_precise(vec v, const int n)
{
  vec s = float32x3_to_oct(v);  // Remap to the square
                                // Each snorm's max value interpreted as an integer,
                                // e.g., 127.0 for snorm8
  float M = float(1 << ((n / 2) - 1)) - 1.0;
  // Remap components to snorm(n/2) precision...with floor instead
  // of round (see equation 1)
  s                        = vec_floor(vec_clamp(s, -1.0f, +1.0f) * M) * (1.0 / M);
  vec   bestRepresentation = s;
  float highestCosine      = vec_dot(oct_to_float32x3(s), v);
  // Test all combinations of floor and ceil and keep the best.
  // Note that at +/- 1, this will exit the square... but that
  // will be a worse encoding and never win.
  for(int i = 0; i <= 1; ++i)
    for(int j = 0; j <= 1; ++j)
      // This branch will be evaluated at compile time
      if((i != 0) || (j != 0))
      {
        // Offset the bit pattern (which is stored in floating
        // point!) to effectively change the rounding mode
        // (when i or j is 0: floor, when it is one: ceiling)
        vec   candidate = vec(i, j, 0) * (1 / M) + s;
        float cosine    = vec_dot(oct_to_float32x3(candidate), v);
        if(cosine > highestCosine)
        {
          bestRepresentation = candidate;
          highestCosine      = cosine;
        }
      }
  return bestRepresentation;
}

//////////////////////////////////////////////////////////////////////////

template <class VertexIndexType>
class Builder
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
    std::vector<MeshletDesc> meshletDescriptors;
  };


  //////////////////////////////////////////////////////////////////////////
  // Builder configuration
private:
  // might want to template these instead of using MAX
  uint32_t m_maxVertexCount;
  uint32_t m_maxPrimitiveCount;

  // due to hw allocation granuarlity, good values are
  // vertex count = 32 or 64
  // primitive count = 40, 84 or 126
  //                   maximizes the fit into gl_PrimitiveIndices[128 * N - 4]
public:
  void setup(uint32_t maxVertexCount, uint32_t maxPrimitiveCount)
  {
    assert(maxPrimitiveCount <= MAX_PRIMITIVE_COUNT_LIMIT);
    assert(maxVertexCount <= MAX_VERTEX_COUNT_LIMIT);

    m_maxVertexCount = maxVertexCount;
    // we may reduce the number of actual triangles a bit to simplify
    // index loader logic in shader. By using less primitives we
    // guarantee to not overshoot the gl_PrimitiveIndices array when using the 32-bit
    // write intrinsic.
    m_maxPrimitiveCount = computePackedPrimitiveCount(maxPrimitiveCount);
  }

  //////////////////////////////////////////////////////////////////////////
  // generate meshlets
private:
  struct PrimitiveCache
  {
    //  Utility class to generate the meshlets from triangle indices.
    //  It finds the unique vertex set used by a series of primitives.
    //  The cache is exhausted if either of the maximums is hit.
    //  The effective limits used with the cache must be < MAX.

    uint8_t  primitives[MAX_PRIMITIVE_COUNT_LIMIT][3];
    uint32_t vertices[MAX_VERTEX_COUNT_LIMIT];
    uint32_t numPrims;
    uint32_t numVertices;

    bool empty() const { return numVertices == 0; }

    void reset()
    {
      numPrims    = 0;
      numVertices = 0;
      // reset
      memset(vertices, 0xFFFFFFFF, sizeof(vertices));
    }

    bool cannotInsert(const VertexIndexType* NV_RESTRICT indices, uint32_t maxVertexSize, uint32_t maxPrimitiveSize) const
    {
      // skip degenerate
      if(indices[0] == indices[1] || indices[0] == indices[2] || indices[1] == indices[2])
      {
        return false;
      }

      uint32_t found = 0;
      for(uint32_t v = 0; v < numVertices; v++)
      {
        for(int i = 0; i < 3; i++)
        {
          uint32_t idx = indices[i];
          if(vertices[v] == idx)
          {
            found++;
          }
        }
      }
      // out of bounds
      return (numVertices + 3 - found) > maxVertexSize || (numPrims + 1) > maxPrimitiveSize;
    }

    void insert(const VertexIndexType* NV_RESTRICT indices)
    {
      uint32_t tri[3];

      // skip degenerate
      if(indices[0] == indices[1] || indices[0] == indices[2] || indices[1] == indices[2])
      {
        return;
      }

      for(int i = 0; i < 3; i++)
      {
        uint32_t idx   = indices[i];
        bool     found = false;
        for(uint32_t v = 0; v < numVertices; v++)
        {
          if(idx == vertices[v])
          {
            tri[i] = v;
            found  = true;
            break;
          }
        }
        if(!found)
        {
          vertices[numVertices] = idx;
          tri[i]                = numVertices;
          numVertices++;
        }
      }

      primitives[numPrims][0] = tri[0];
      primitives[numPrims][1] = tri[1];
      primitives[numPrims][2] = tri[2];
      numPrims++;
    }
  };

  void addMeshlet(MeshletGeometry& geometry, const PrimitiveCache& cache) const
  {
    MeshletDesc meshlet;
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
      if(PRIMITIVE_PACKING == NVMESHLET_PACKING_TRIANGLE_UINT32)
      {
        geometry.primitiveIndices.push_back(cache.primitives[p][2]);
      }
    }

    while((geometry.vertexIndices.size() % VERTEX_PACKING_ALIGNMENT) != 0)
    {
      geometry.vertexIndices.push_back(cache.vertices[cache.numVertices - 1]);
    }
    size_t idx = 0;
    while((geometry.primitiveIndices.size() % PRIMITIVE_PACKING_ALIGNMENT) != 0)
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
    cache.reset();

    for(uint32_t i = 0; i < numIndices / 3; i++)
    {
      if(cache.cannotInsert(indices + i * 3, m_maxVertexCount, m_maxPrimitiveCount))
      {
        // finish old and reset
        addMeshlet(geometry, cache);
        cache.reset();

        // if we exhausted the index buffers, return early
        if(!MeshletDesc::isPrimBeginLegal(uint32_t(geometry.primitiveIndices.size()))
           || !MeshletDesc::isVertexBeginLegal(uint32_t(geometry.vertexIndices.size())))
        {
          return i * 3;
        }
      }
      cache.insert(indices + i * 3);
    }
    if(!cache.empty())
    {
      addMeshlet(geometry, cache);
    }

    return numIndices;
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

    for(size_t i = 0; i < geometry.meshletDescriptors.size(); i++)
    {
      MeshletDesc& meshlet = geometry.meshletDescriptors[i];

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
        const uint32_t primStride = (PRIMITIVE_PACKING == NVMESHLET_PACKING_TRIANGLE_UINT32) ? 4 : 3;

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

  enum StatusCode
  {
    STATUS_NO_ERROR,
    STATUS_PRIM_OUT_OF_BOUNDS,
    STATUS_VERTEX_OUT_OF_BOUNDS,
    STATUS_MISMATCH_INDICES,
  };

  StatusCode errorCheck(const MeshletGeometry& geometry,
                        uint32_t               minVertex,
                        uint32_t               maxVertex,
                        uint32_t               numIndices,
                        const VertexIndexType* NV_RESTRICT indices) const
  {
    uint32_t compareTris = 0;

    for(size_t i = 0; i < geometry.meshletDescriptors.size(); i++)
    {
      const MeshletDesc& meshlet = geometry.meshletDescriptors[i];

      uint32_t primCount   = meshlet.getNumPrims();
      uint32_t vertexCount = meshlet.getNumVertices();

      uint32_t primBegin   = meshlet.getPrimBegin();
      uint32_t vertexBegin = meshlet.getVertexBegin();

      // skip unset
      if(vertexCount == 1)
        continue;

      for(uint32_t p = 0; p < primCount; p++)
      {
        const uint32_t primStride = (PRIMITIVE_PACKING == NVMESHLET_PACKING_TRIANGLE_UINT32) ? 4 : 3;

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
      const MeshletDesc& meshlet     = geometry.meshletDescriptors[i];
      uint32_t           primCount   = meshlet.getNumPrims();
      uint32_t           vertexCount = meshlet.getNumVertices();

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
      const MeshletDesc& meshlet     = geometry.meshletDescriptors[i];
      uint32_t           primCount   = meshlet.getNumPrims();
      uint32_t           vertexCount = meshlet.getNumVertices();
      double             diff;

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
