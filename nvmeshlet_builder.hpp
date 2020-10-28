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

#ifndef _NV_MESHLET_BUILDER_H__
#define _NV_MESHLET_BUILDER_H__

#include <NvFoundation.h>
#include <algorithm>
#include <cstdint>
#include <vector>
#include <stdio.h>

#ifdef _MSC_VER
#include <intrin.h>
#endif

namespace NVMeshlet {
// Each Meshlet can have a varying count of its maximum number
// of vertices and primitives. We hardcode a few absolute maxima
// to accellerate some of the functions and allow usage of
// smaller datastructures.

// The builder, however, is configurable to use smaller maxima,
// which is recommended.

// The limits below are hard limits due to the encoding chosen for the
// meshlet descriptor. Actual hw-limits can be higher, but typically
// make things slower due to large on-chip allocation.

#define NVMESHLET_ASSERT_ON_DEGENERATES      1

static const int MAX_VERTEX_COUNT_LIMIT    = 256;
static const int MAX_PRIMITIVE_COUNT_LIMIT = 256;

static const uint32_t MESHLETS_PER_TASK = 32;


// must not change
typedef uint8_t PrimitiveIndexType;  // must store [0,MAX_VERTEX_COUNT_LIMIT-1]

inline uint32_t computeTasksCount(uint32_t numMeshlets)
{
  return (numMeshlets + MESHLETS_PER_TASK - 1) / MESHLETS_PER_TASK;
}

inline uint32_t alignedSize(uint32_t v, uint32_t align) {
  return (v + align - 1) & (~(align-1));
}


// opaque type, all builders will specialize this, but fit within
struct MeshletDesc
{
  uint32_t fieldX;
  uint32_t fieldY;
  uint32_t fieldZ;
  uint32_t fieldW;
};

struct MeshletBbox
{
  float bboxMin[3];
  float bboxMax[3];

  MeshletBbox() {
    bboxMin[0] = FLT_MAX;
    bboxMin[1] = FLT_MAX;
    bboxMin[2] = FLT_MAX;
    bboxMax[0] = -FLT_MAX;
    bboxMax[1] = -FLT_MAX;
    bboxMax[2] = -FLT_MAX;
  }
};

enum StatusCode
{
  STATUS_NO_ERROR,
  STATUS_PRIM_OUT_OF_BOUNDS,
  STATUS_VERTEX_OUT_OF_BOUNDS,
  STATUS_MISMATCH_INDICES,
};

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

  size_t posBitTotal  = 0;

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
            "meshlets; %7zd; prim; %9zd; %.2f; vertex; %9zd; %.2f; backface; %.2f; waste; v; %.2f; p; %.2f; m; %.2f;\n", meshletsTotal,
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

// quantized vector
struct qvec 
{
  uint32_t bits[3];

  qvec()
  {
    bits[0] = 0;
    bits[1] = 0;
    bits[2] = 0;
  }
  qvec(uint32_t raw)
  {
    bits[0] = raw;
    bits[1] = raw;
    bits[2] = raw;
  }
  qvec(uint32_t x, uint32_t y, uint32_t z)
  {
    bits[0] = x;
    bits[1] = y;
    bits[2] = z;
  }
  qvec(const vec& v, const vec& bboxMin, const vec& bboxExtent, float quantizedMul)
  {
    vec nrm = (v - bboxMin) / bboxExtent;
    bits[0]  = uint32_t(round(nrm.x * quantizedMul));
    bits[1]  = uint32_t(round(nrm.y * quantizedMul));
    bits[2]  = uint32_t(round(nrm.z * quantizedMul));
  }
};

inline qvec operator-(const qvec& a, const qvec& b)
{
  return qvec(a.bits[0]- b.bits[0], a.bits[1] - b.bits[1], a.bits[2] - b.bits[2]);
}

inline qvec qvec_min(const qvec& a, const qvec& b)
{
  return qvec(std::min(a.bits[0], b.bits[0]), std::min(a.bits[1], b.bits[1]), std::min(a.bits[2], b.bits[2]));
}

inline qvec qvec_max(const qvec& a, const qvec& b)
{
  return qvec(std::max(a.bits[0], b.bits[0]), std::max(a.bits[1], b.bits[1]), std::max(a.bits[2], b.bits[2]));
}

//////////////////////////////////////////////////////////////////////////

inline uint32_t pack(uint32_t value, int width, int offset)
{
  return (uint32_t)((value & ((1 << width) - 1)) << offset);
}
inline uint32_t unpack(uint32_t value, int width, int offset)
{
  return (uint32_t)((value >> offset) & ((1 << width) - 1));
}

inline void setBitField(uint32_t arraySize, uint32_t* bits, uint32_t width, uint32_t offset, uint32_t value)
{
  uint32_t idx     = offset / 32u;
  uint32_t shiftLo = offset % 32;

  assert(idx < arraySize);

  bool onlyLo = (shiftLo + width) <= 32;

  uint32_t sizeLo = onlyLo ? width : 32 - shiftLo;
  uint32_t sizeHi = onlyLo ? 0 : (shiftLo + width - 32);

  uint32_t shiftHi = sizeLo;

  uint32_t retLo = (value << shiftLo);
  uint32_t retHi = (value >> shiftHi);

  bits[idx] |= retLo;
  if(idx + 1 < arraySize)
  {
    bits[idx + 1] |= retHi;
  }
}

inline uint32_t getBitField(uint32_t arraySize, const uint32_t* bits, uint32_t width, uint32_t offset)
{
  uint32_t idx = offset / 32;

  // assumes out-of-bounds access is not fatal
  uint32_t rawLo = bits[idx];
  uint32_t rawHi = idx + 1 < arraySize ? bits[idx + 1] : 0;

  uint32_t shiftLo = offset % 32;

  bool onlyLo = (shiftLo + width) <= 32;

  uint32_t sizeLo = onlyLo ? width : 32 - shiftLo;
  uint32_t sizeHi = onlyLo ? 0 : (shiftLo + width - 32);

  uint32_t shiftHi = sizeLo;

  uint32_t maskLo = (width == 32) ? ~0 : ((1 << sizeLo) - 1);
  uint32_t maskU  = (1 << sizeHi) - 1;

  uint32_t retLo = ((rawLo >> shiftLo) & maskLo);
  uint32_t retHi = ((rawHi & maskU) << shiftHi);

  return retLo | retHi;
}

#if defined(_MSC_VER)

#pragma intrinsic(_BitScanReverse)

inline uint32_t findMSB(uint32_t value)
{
  unsigned long idx = 0;
  _BitScanReverse(&idx, value);
  return idx;
}
#else
inline uint32_t findMSB(uint32_t value)
{
  uint32_t idx = __builtin_clz(value);
  return idx;
}
#endif

//////////////////////////////////////////////////////////////////////////

struct PrimitiveCache
{
  //  Utility class to generate the meshlets from triangle indices.
  //  It finds the unique vertex set used by a series of primitives.
  //  The cache is exhausted if either of the maximums is hit.
  //  The effective limits used with the cache must be < MAX.

  PrimitiveIndexType  primitives[MAX_PRIMITIVE_COUNT_LIMIT][3];
  uint32_t vertices[MAX_VERTEX_COUNT_LIMIT];
  uint32_t numPrims;
  uint32_t numVertices;
  uint32_t numVertexDeltaBits;
  uint32_t numVertexAllBits;

  uint32_t maxVertexSize;
  uint32_t maxPrimitiveSize;
  uint32_t primitiveBits = 1;
  uint32_t maxBlockBits  = ~0;

  bool empty() const { return numVertices == 0; }

  void reset()
  {
    numPrims           = 0;
    numVertices        = 0;
    numVertexDeltaBits = 0;
    numVertexAllBits   = 0;
    // reset
    memset(vertices, 0xFFFFFFFF, sizeof(vertices));
  }

  bool fitsBlock() const
  {
    uint32_t primBits = (numPrims - 1) * 3 * primitiveBits;
    uint32_t vertBits = (numVertices - 1) * numVertexDeltaBits;
    bool     state    = (primBits + vertBits) <= maxBlockBits;
    return state;
  }

  bool cannotInsert(uint32_t idxA, uint32_t idxB, uint32_t idxC) const
  {
    const uint32_t indices[3] = {idxA, idxB, idxC};
    // skip degenerate
    if(indices[0] == indices[1] || indices[0] == indices[2] || indices[1] == indices[2])
    {
#if NVMESHLET_ASSERT_ON_DEGENERATES
      //assert(0 && "degenerate triangle");
#endif
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

  bool cannotInsertBlock(uint32_t idxA, uint32_t idxB, uint32_t idxC) const
  {
    const uint32_t indices[3] = {idxA, idxB, idxC};
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
    // ensure one bit is set in deltas for findMSB returning 0
    uint32_t firstVertex = numVertices ? vertices[0] : indices[0];
    uint32_t cmpBits     = std::max(findMSB((firstVertex ^ indices[0]) | 1),
                                std::max(findMSB((firstVertex ^ indices[1]) | 1), findMSB((firstVertex ^ indices[2]) | 1)))
                       + 1;

    uint32_t deltaBits = std::max(cmpBits, numVertexDeltaBits);

    uint32_t newVertices = numVertices + 3 - found;
    uint32_t newPrims    = numPrims + 1;

    uint32_t newBits;

    {
      uint32_t newVertBits = (newVertices - 1) * deltaBits;
      uint32_t newPrimBits = (newPrims - 1) * 3 * primitiveBits;
      newBits              = newVertBits + newPrimBits;
    }

    // out of bounds
    return (newPrims > maxPrimitiveSize) || (newVertices > maxVertexSize) || (newBits > maxBlockBits);
  }

  void insert(uint32_t idxA, uint32_t idxB, uint32_t idxC)
  {
    const uint32_t indices[3] = {idxA, idxB, idxC};
    uint32_t       tri[3];

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

        if(numVertices)
        {
          numVertexDeltaBits = std::max(findMSB((idx ^ vertices[0]) | 1) + 1, numVertexDeltaBits);
        }
        numVertexAllBits = std::max(numVertexAllBits, findMSB(idx) + 1);

        numVertices++;
      }
    }

    primitives[numPrims][0] = tri[0];
    primitives[numPrims][1] = tri[1];
    primitives[numPrims][2] = tri[2];
    numPrims++;

    assert(fitsBlock());
  }
};

}  // namespace NVMeshlet

#endif
