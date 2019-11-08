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


#include "cadscene.hpp"
#include <fileformats/cadscenefile.h>

#include "config.h"
#include "nvmeshlet_builder.hpp"
#include <nvh/geometry.hpp>
#include <nvh/misc.hpp>

#include <algorithm>
#include <assert.h>
#include <platform.h>

#define USE_CACHECOMBINE 1

NV_INLINE half floatToHalf(float fval)
{
  unsigned long ival = *(unsigned long*)(&fval);
  if(!ival)
  {
    return 0;
  }
  else
  {
    int e = ((ival & 0x7f800000) >> 23) - 127 + 15;
    if(e < 0)
    {
      return 0;
    }
    else if(e > 31)
    {
      e = 31;
    }
    {
      unsigned long s = ival & 0x80000000;
      unsigned long f = ival & 0x007fffff;
      return (half)(((s >> 16) & 0x8000) | ((e << 10) & 0x7c00) | ((f >> 13) & 0x03ff));
    }
  }
}

NV_INLINE void floatToHalfVector(half output[3], const nvmath::vec3f& input)
{
  output[0] = floatToHalf(input[0]);
  output[1] = floatToHalf(input[1]);
  output[2] = floatToHalf(input[2]);
}

NV_INLINE void floatToHalfVector(half output[4], const nvmath::vec4f& input)
{
  output[0] = floatToHalf(input[0]);
  output[1] = floatToHalf(input[1]);
  output[2] = floatToHalf(input[2]);
  output[3] = floatToHalf(input[3]);
}

nvmath::vec4f randomVector(float from, float to)
{
  nvmath::vec4f vec;
  float         width = to - from;
  for(int i = 0; i < 4; i++)
  {
    vec.vec_array[i] = from + (float(rand()) / float(RAND_MAX)) * width;
  }
  return vec;
}

static void storeData(std::vector<uint32_t>& container, size_t size, const void* data)
{
  container.resize((size + 3) / 4, 0);
  memcpy(container.data(), data, size);
}

bool CadScene::loadCSF(const char* filename, const LoadConfig& cfg, int clones, int cloneaxis)
{
  CSFile* csf;

  if(!m_geometry.empty())
    return false;


  CSFileMemoryPTR csfmem = CSFileMemory_new();

  if(CSFile_loadExt(&csf, filename, csfmem) != CADSCENEFILE_NOERROR
     || !(csf->fileFlags & (CADSCENEFILE_FLAG_UNIQUENODES | CADSCENEFILE_FLAG_STRIPS)))
  {
    CSFileMemory_delete(csfmem);
    return false;
  }

  m_cfg = cfg;

  int copies = clones + 1;

  {
    // propagate scale onto matrix tree
    csf->nodes[csf->rootIDX].objectTM[0] *= cfg.scale;
    csf->nodes[csf->rootIDX].objectTM[5] *= cfg.scale;
    csf->nodes[csf->rootIDX].objectTM[10] *= cfg.scale;

    CSFile_transform(csf);
  }

  srand(234525);

  // materials
  m_materials.resize(csf->numMaterials);
  for(int n = 0; n < csf->numMaterials; n++)
  {
    CSFMaterial* csfmaterial = &csf->materials[n];
    Material&    material    = m_materials[n];

    for(int i = 0; i < 2; i++)
    {
      material.sides[i].ambient  = randomVector(0.0f, 0.1f);
      material.sides[i].diffuse  = nvmath::vec4f(csf->materials[n].color) + randomVector(0.0f, 0.07f);
      material.sides[i].specular = randomVector(0.25f, 0.55f);
      material.sides[i].emissive = randomVector(0.0f, 0.05f);
    }
  }

  int tshorts = 0;
  int ttotal  = 0;

  m_vboSize  = 0;
  m_iboSize  = 0;
  m_meshSize = 0;

  // geometry
  int numGeoms = csf->numGeometries;
  m_geometry.resize(csf->numGeometries);

  int numBboxes    = csf->numGeometries;
  int numGeomParts = 0;
  for(int g = 0; g < csf->numGeometries; g++)
  {
    CSFGeometry* csfgeom = &csf->geometries[g];
    Geometry&    geom    = m_geometry[g];

    geom.partOffset     = numGeomParts;
    geom.partBboxOffset = numBboxes;
    numBboxes += csfgeom->numParts;
    numGeomParts += csfgeom->numParts;
  }

  m_numGeometryParts = numGeomParts;

  m_bboxes.resize(numBboxes);

#pragma omp parallel for
  for(int g = 0; g < csf->numGeometries; g++)
  {
    CSFGeometry* csfgeom = &csf->geometries[g];
    Geometry&    geom    = m_geometry[g];

    geom.numVertices   = csfgeom->numVertices;
    geom.numIndexSolid = csfgeom->numIndexSolid;

    geom.useShorts = geom.numVertices < 0xFFFF && m_cfg.allowShorts;
    geom.meshSize  = 0;

    geom.vboSize = getVertexSize() * csfgeom->numVertices;
    geom.aboSize = getVertexAttributeSize() * csfgeom->numVertices;

    geom.vboData = malloc(geom.vboSize);
    geom.aboData = malloc(geom.aboSize);

    memset(geom.aboData, 0, geom.aboSize);

    const bool colorizeExtra = true;

    if(m_cfg.fp16)
    {
      for(int i = 0; i < csfgeom->numVertices; i++)
      {
        VertexFP16*           vertex    = (VertexFP16*)getVertex(geom.vboData, i);
        VertexAttributesFP16* attribute = (VertexAttributesFP16*)getVertexAttribute(geom.aboData, i);

        nvmath::vec4f position;
        nvmath::vec4f normal;
        position[0] = csfgeom->vertex[3 * i + 0];
        position[1] = csfgeom->vertex[3 * i + 1];
        position[2] = csfgeom->vertex[3 * i + 2];
        position[3] = 1.0f;
        normal[0]   = csfgeom->normal[3 * i + 0];
        normal[1]   = csfgeom->normal[3 * i + 1];
        normal[2]   = csfgeom->normal[3 * i + 2];
        normal[3]   = 0.0f;

        floatToHalfVector(vertex->position, position);
        floatToHalfVector(attribute->normal, normal);

        for(uint32_t i = 0; m_cfg.colorizeExtra && i < m_cfg.extraAttributes; i++)
        {
          floatToHalfVector(attribute[1 + i].normal, nvmath::vec4f(0, 1, 0, 0) * 0.1f);
        }

        m_bboxes[g].merge(position);
      }
    }
    else
    {
      for(int i = 0; i < csfgeom->numVertices; i++)
      {
        Vertex*           vertex    = (Vertex*)getVertex(geom.vboData, i);
        VertexAttributes* attribute = (VertexAttributes*)getVertexAttribute(geom.aboData, i);

        nvmath::vec4f position;
        nvmath::vec4f normal;
        position[0] = csfgeom->vertex[3 * i + 0];
        position[1] = csfgeom->vertex[3 * i + 1];
        position[2] = csfgeom->vertex[3 * i + 2];
        position[3] = 1.0f;
        normal[0]   = csfgeom->normal[3 * i + 0];
        normal[1]   = csfgeom->normal[3 * i + 1];
        normal[2]   = csfgeom->normal[3 * i + 2];
        normal[3]   = 0.0f;

        vertex->position  = position;
        attribute->normal = normal;

        for(uint32_t i = 0; m_cfg.colorizeExtra && i < m_cfg.extraAttributes; i++)
        {
          attribute[1 + i].normal = nvmath::vec4f(0, 1, 0, 0) * 0.1f;
        }

        m_bboxes[g].merge(position);
      }
    }


    size_t indexSize = 0;
    if(geom.useShorts)
    {
      indexSize    = sizeof(uint16_t);
      geom.iboSize = sizeof(uint16_t) * csfgeom->numIndexSolid;

      uint16_t* indices = (uint16_t*)malloc(geom.iboSize);
      for(int i = 0; i < csfgeom->numIndexSolid; i++)
      {
        indices[i] = csfgeom->indexSolid[i];
      }

      geom.iboData = indices;
    }
    else
    {
      indexSize    = sizeof(uint32_t);
      geom.iboSize = csfgeom->numIndexSolid * sizeof(uint32_t);

      uint32_t* indices = (uint32_t*)malloc(geom.iboSize);
      memcpy(indices, csfgeom->indexSolid, geom.iboSize);

      geom.iboData = indices;
    }

    geom.parts.resize(csfgeom->numParts);

    uint32_t accumSolid  = 0;
    size_t   offsetSolid = 0;
    for(int p = 0; p < csfgeom->numParts; p++)
    {
      geom.parts[p].indexSolid.count  = csfgeom->parts[p].numIndexSolid;
      geom.parts[p].indexSolid.offset = offsetSolid;

      for(int i = 0; i < csfgeom->parts[p].numIndexSolid; i++)
      {
        uint32_t v = csfgeom->indexSolid[i + accumSolid];

        nvmath::vec4f position;
        position[0] = csfgeom->vertex[3 * v + 0];
        position[1] = csfgeom->vertex[3 * v + 1];
        position[2] = csfgeom->vertex[3 * v + 2];
        position[3] = 1.0f;

        m_bboxes[geom.partBboxOffset + p].merge(position);
      }

      offsetSolid += csfgeom->parts[p].numIndexSolid * indexSize;
      accumSolid += csfgeom->parts[p].numIndexSolid;
    }

#pragma omp critical
    {
      tshorts += geom.useShorts;
      ttotal++;

      m_vboSize += geom.vboSize + geom.aboSize;
      m_iboSize += geom.iboSize;
    }
  }

  LOGI("geometries: shorts %d, total %d\n", tshorts, ttotal);


  srand(63546);
  std::vector<nvmath::vec4f> geometryColors(csf->numGeometries);
  for(int g = 0; g < csf->numGeometries; g++)
  {
    geometryColors[g] = nvmath::vec4f(nvh::frand(), nvh::frand(), nvh::frand(), 1.0f);
  }

  // nodes
  int numObjects = 0;
  m_matrices.resize(csf->numNodes * copies);
  for(int n = 0; n < csf->numNodes; n++)
  {
    CSFNode* csfnode = &csf->nodes[n];

    memcpy(m_matrices[n].objectMatrix.get_value(), csfnode->objectTM, sizeof(float) * 16);
    memcpy(m_matrices[n].worldMatrix.get_value(), csfnode->worldTM, sizeof(float) * 16);

    m_matrices[n].worldMatrixIT = nvmath::transpose(nvmath::invert(m_matrices[n].worldMatrix));

    if(csfnode->geometryIDX < 0)
      continue;

    m_matrices[n].winding = nvmath::det(m_matrices[n].worldMatrix) > 0 ? 1.0f : -1.0f;
    m_matrices[n].bboxMin = m_bboxes[csfnode->geometryIDX].min;
    m_matrices[n].bboxMax = m_bboxes[csfnode->geometryIDX].max;

    if(true && csfnode->numParts)
    {
      m_matrices[n].color = m_materials[csfnode->parts[0].materialIDX].sides[0].diffuse;
    }
    else
    {
      m_matrices[n].color = geometryColors[csfnode->geometryIDX];
    }

    numObjects++;
  }


  // objects
  m_objects.resize(numObjects * copies);
  numObjects   = 0;
  int numParts = 0;
  for(int n = 0; n < csf->numNodes; n++)
  {
    CSFNode* csfnode = &csf->nodes[n];

    if(csfnode->geometryIDX < 0)
      continue;

    Object& object = m_objects[numObjects];

    object.partOffset    = numParts;
    object.matrixIndex   = n;
    object.geometryIndex = csfnode->geometryIDX;

    object.faceCCW = nvmath::det(m_matrices[object.matrixIndex].worldMatrix) > 0;

    object.parts.resize(csfnode->numParts);
    for(int i = 0; i < csfnode->numParts; i++)
    {
      object.parts[i].active        = csfnode->parts[i].active ? 1 : 0;
      object.parts[i].matrixIndex   = n;
      object.parts[i].materialIndex = csfnode->parts[i].materialIDX;
#if 1
      if(csf->materials[csfnode->parts[i].materialIDX].color[3] < 0.9f)
      {
        object.parts[i].active = 0;
      }
#endif
    }

    BBox bbox = m_bboxes[object.geometryIndex].transformed(m_matrices[object.matrixIndex].worldMatrix);
    m_bbox.merge(bbox);

    numObjects++;
    numParts += csfnode->numParts;
  }
  m_numObjectParts = numParts;

  // compute clone move delta based on m_bbox;

  nvmath::vec4f dim = m_bbox.max - m_bbox.min;
  m_bboxInstanced   = m_bbox;

  int sq      = 1;
  int numAxis = 0;
  for(int i = 0; i < 3; i++)
  {
    numAxis += (cloneaxis & (1 << i)) ? 1 : 0;
  }

  assert(numAxis);

  switch(numAxis)
  {
    case 1:
      sq = copies;
      break;
    case 2:
      while(sq * sq < copies)
      {
        sq++;
      }
      break;
    case 3:
      while(sq * sq * sq < copies)
      {
        sq++;
      }
      break;
  }


  for(int c = 1; c <= clones; c++)
  {
    int numNodes = csf->numNodes;

    nvmath::vec4f shift = dim * 1.05f;

    float u = 0;
    float v = 0;
    float w = 0;

    switch(numAxis)
    {
      case 1:
        u = float(c);
        break;
      case 2:
        u = float(c % sq);
        v = float(c / sq);
        break;
      case 3:
        u = float(c % sq);
        v = float((c / sq) % sq);
        w = float(c / (sq * sq));
        break;
    }

    float use = u;

    if(cloneaxis & (1 << 0))
    {
      shift.x *= -use;
      if(numAxis > 1)
        use = v;
    }
    else
    {
      shift.x = 0;
    }

    if(cloneaxis & (1 << 1))
    {
      shift.y *= use;
      if(numAxis > 2)
        use = w;
      else if(numAxis > 1)
        use = v;
    }
    else
    {
      shift.y = 0;
    }

    if(cloneaxis & (1 << 2))
    {
      shift.z *= -use;
    }
    else
    {
      shift.z = 0;
    }

    shift.w = 0;

    // move all world matrices
    for(int n = 0; n < numNodes; n++)
    {
      MatrixNode& node = m_matrices[n + numNodes * c];
      node             = m_matrices[n];
      node.worldMatrix.set_col(3, node.worldMatrix.col(3) + shift);
    }

    {
      // patch object matrix of root
      MatrixNode& node = m_matrices[csf->rootIDX + numNodes * c];
      node.objectMatrix.set_col(3, node.objectMatrix.col(3) + shift);
    }

    // clone objects
    for(int n = 0; n < numObjects; n++)
    {
      const Object& objectorig = m_objects[n];
      Object&       object     = m_objects[n + numObjects * c];

      object = objectorig;
      object.matrixIndex += c * numNodes;
      for(size_t p = 0; p < object.parts.size(); p++)
      {
        object.parts[p].matrixIndex += c * numNodes;
      }

      BBox bbox = m_bboxes[object.geometryIndex].transformed(m_matrices[object.matrixIndex].worldMatrix);
      m_bboxInstanced.merge(bbox);
    }
  }

  if(cfg.meshPrimitiveCount && cfg.meshVertexCount)
  {
    buildMeshletTopology(csf);
  }

  CSFileMemory_delete(csfmem);

  return true;
}

void CadScene::unload()
{
  if(m_geometry.empty())
    return;

  m_matrices.clear();
  m_materials.clear();
  m_geometry.clear();
  m_objects.clear();
  m_bboxes.clear();
}


size_t fillIndexBuffer(int useShorts, const std::vector<unsigned int>& vertexindices, void*& storage)
{
  size_t vidxSize;

  if(useShorts)
  {
    vidxSize = sizeof(uint16_t) * vertexindices.size();

    uint16_t* vertexindices16 = (uint16_t*)malloc(vidxSize);
    for(size_t i = 0; i < vertexindices.size(); i++)
    {
      vertexindices16[i] = vertexindices[i];
    }

    storage = vertexindices16;
  }
  else
  {
    vidxSize = sizeof(uint32_t) * vertexindices.size();

    uint32_t* vertexindices32 = (uint32_t*)malloc(vidxSize);
    memcpy(vertexindices32, vertexindices.data(), vidxSize);

    storage = vertexindices32;
  }

  return vidxSize;
}

typedef NVMeshlet::Builder<uint32_t> MeshletBuilder;

void fillMeshletTopology(MeshletBuilder::MeshletGeometry& geometry, CadScene::MeshletTopology& topo, int useShorts)
{
  if(geometry.meshletDescriptors.empty())
    return;

  topo.vertSize = fillIndexBuffer(useShorts, geometry.vertexIndices, topo.vertData);

  topo.descSize = sizeof(NVMeshlet::MeshletDesc) * geometry.meshletDescriptors.size();
  topo.primSize = sizeof(NVMeshlet::PrimitiveIndexType) * geometry.primitiveIndices.size();

  topo.descData = malloc(topo.descSize);
  topo.primData = malloc(topo.primSize);

  memcpy(topo.descData, geometry.meshletDescriptors.data(), geometry.meshletDescriptors.size() * sizeof(NVMeshlet::MeshletDesc));
  memcpy(topo.primData, geometry.primitiveIndices.data(), topo.primSize);
}

void CadScene::buildMeshletTopology(const CSFile* csf)
{
  NVMeshlet::Builder<uint32_t> meshletBuilder;
  meshletBuilder.setup(m_cfg.meshVertexCount, m_cfg.meshPrimitiveCount);

  NVMeshlet::Stats statsGlobal;

  uint32_t groups = 0;

#pragma omp parallel for
  for(int g = 0; g < csf->numGeometries; g++)
  {
    const CSFGeometry* geo  = csf->geometries + g;
    Geometry&          geom = m_geometry[g];

    NVMeshlet::Builder<uint32_t>::MeshletGeometry meshletGeometry;

    uint32_t numMeshlets = 0;
    uint32_t indexOffset = 0;
    for(int p = 0; p < geo->numParts; p++)
    {
      uint32_t numIndex = geo->parts[p].numIndexSolid;

      geom.parts[p].meshSolid.offset = numMeshlets;

      uint32_t processedIndices = meshletBuilder.buildMeshlets(meshletGeometry, numIndex, geo->indexSolid + indexOffset);
      if(processedIndices != numIndex)
      {
        LOGE("warning: geometry meshlet incomplete %d\n", g);
      }

      geom.parts[p].meshSolid.count = (uint32_t)meshletGeometry.meshletDescriptors.size() - numMeshlets;
      numMeshlets                   = (uint32_t)meshletGeometry.meshletDescriptors.size();
      indexOffset += numIndex;
    }

    geom.meshlet.numMeshlets = int(meshletGeometry.meshletDescriptors.size());

    meshletBuilder.buildMeshletEarlyCulling(meshletGeometry, m_bboxes[g].min.vec_array, m_bboxes[g].max.vec_array,
                                            geo->vertex, sizeof(float) * 3);
    if(m_cfg.verbose)
    {
#if 0
      int errorcode = meshletBuilder.validate(meshletGeometries[g], 0, geo->numVertices - 1, geo->numIndexSolid, geo->indexSolid);
      if (errorcode) {
        LOGE("geometry %d: meshlet error %d\n", g, errorcode);
      }
#endif

      NVMeshlet::Stats statsLocal;
      meshletBuilder.appendStats(meshletGeometry, statsLocal);

#pragma omp critical
      {
        statsGlobal.append(statsLocal);
      }
    }

    fillMeshletTopology(meshletGeometry, geom.meshlet, geom.useShorts);

    geom.meshSize = NVMeshlet::computeCommonAlignedSize(geom.meshlet.descSize)
                    + NVMeshlet::computeCommonAlignedSize(geom.meshlet.primSize)
                    + NVMeshlet::computeCommonAlignedSize(geom.meshlet.vertSize);

#pragma omp critical
    {
      m_meshSize += geom.meshSize;
      groups += numMeshlets;
    }
  }

  LOGI("meshlet config: %d vertices, %d primitives\n", m_cfg.meshVertexCount, m_cfg.meshPrimitiveCount);

  if(m_cfg.verbose)
  {
    statsGlobal.fprint(stdout);
  }

  LOGI("meshlet total: %d\n", groups);
}
