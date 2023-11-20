/*
 * Copyright (c) 2017-2023, NVIDIA CORPORATION.  All rights reserved.
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


#include "cadscene.hpp"
#include <fileformats/cadscenefile.h>

#include "config.h"
#include "nvmeshlet_packbasic.hpp"
#include <nvh/geometry.hpp>
#include <nvh/misc.hpp>

#include <algorithm>
#include <cassert>
#include <platform.h>
#include <glm/gtc/type_ptr.hpp>

NV_INLINE half floatToHalf(float fval)
{
  unsigned long ival = *(unsigned long*)(&fval);
  if(!ival)
  {
    return 0;
  }
  else
  {
    unsigned long e = ((ival & 0x7f800000) >> 23) - 127 + 15;
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

NV_INLINE void floatToHalfVector(half output[3], const glm::vec3& input)
{
  output[0] = floatToHalf(input[0]);
  output[1] = floatToHalf(input[1]);
  output[2] = floatToHalf(input[2]);
}

NV_INLINE void floatToHalfVector(half output[4], const glm::vec4& input)
{
  output[0] = floatToHalf(input[0]);
  output[1] = floatToHalf(input[1]);
  output[2] = floatToHalf(input[2]);
  output[3] = floatToHalf(input[3]);
}

glm::vec4 randomVector(float from, float to)
{
  glm::vec4 vec;
  float     width = to - from;
  for(int i = 0; i < 4; i++)
  {
    vec[i] = from + (float(rand()) / float(RAND_MAX)) * width;
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
    Material& material = m_materials[n];

    for(auto& side : material.sides)
    {
      side.ambient  = randomVector(0.0f, 0.1f);
      side.diffuse  = glm::make_vec4(csf->materials[n].color) + randomVector(0.0f, 0.07f);
      side.specular = randomVector(0.25f, 0.55f);
      side.emissive = randomVector(0.0f, 0.05f);
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
      for(uint32_t i = 0; i < uint32_t(csfgeom->numVertices); i++)
      {
        VertexFP16*           vertex    = (VertexFP16*)getVertex(geom.vboData, i);
        VertexAttributesFP16* attribute = (VertexAttributesFP16*)getVertexAttribute(geom.aboData, i);

        glm::vec4 position;
        glm::vec4 normal;
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
          floatToHalfVector(attribute[1 + i].normal, glm::vec4(0, 1, 0, 0) * 0.1f);
        }

        m_bboxes[g].merge(position);
      }
    }
    else
    {
      for(uint32_t i = 0; i < uint32_t(csfgeom->numVertices); i++)
      {
        Vertex*           vertex    = (Vertex*)getVertex(geom.vboData, i);
        VertexAttributes* attribute = (VertexAttributes*)getVertexAttribute(geom.aboData, i);

        glm::vec4 position;
        glm::vec4 normal;
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
          attribute[1 + i].normal = glm::vec4(0, 1, 0, 0) * 0.1f;
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
      for(uint32_t i = 0; i < uint32_t(csfgeom->numIndexSolid); i++)
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
    for(uint32_t p = 0; p < uint32_t(csfgeom->numParts); p++)
    {
      geom.parts[p].indexSolid.count  = csfgeom->parts[p].numIndexSolid;
      geom.parts[p].indexSolid.offset = offsetSolid;

      for(uint32_t i = 0; i < uint32_t(csfgeom->parts[p].numIndexSolid); i++)
      {
        uint32_t v = csfgeom->indexSolid[i + accumSolid];

        glm::vec4 position;
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

  LOGI("geometries: shorts %d, total %d\n", tshorts, ttotal)


  srand(63546);
  std::vector<glm::vec4> geometryColors(csf->numGeometries);
  for(int g = 0; g < csf->numGeometries; g++)
  {
    geometryColors[g] = glm::vec4(nvh::frand(), nvh::frand(), nvh::frand(), 1.0f);
  }

  // nodes
  int numObjects = 0;
  m_matrices.resize(csf->numNodes * copies);
  for(int n = 0; n < csf->numNodes; n++)
  {
    CSFNode* csfnode = &csf->nodes[n];

    memcpy(glm::value_ptr(m_matrices[n].objectMatrix), csfnode->objectTM, sizeof(float) * 16);
    memcpy(glm::value_ptr(m_matrices[n].worldMatrix), csfnode->worldTM, sizeof(float) * 16);

    m_matrices[n].worldMatrixIT = glm::transpose(glm::inverse(m_matrices[n].worldMatrix));

    if(csfnode->geometryIDX < 0)
      continue;

    m_matrices[n].winding = glm::determinant(m_matrices[n].worldMatrix) > 0 ? 1.0f : -1.0f;
    m_matrices[n].bboxMin = m_bboxes[csfnode->geometryIDX].min;
    m_matrices[n].bboxMax = m_bboxes[csfnode->geometryIDX].max;

    if(csfnode->numParts != 0)
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

    object.faceCCW = glm::determinant(m_matrices[object.matrixIndex].worldMatrix) > 0;

    object.parts.resize(csfnode->numParts);
    for(uint32_t i = 0; i < uint32_t(csfnode->numParts); i++)
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

  glm::vec4 dim   = m_bbox.max - m_bbox.min;
  m_bboxInstanced = m_bbox;

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
    default:
      assert(false);
  }


  for(int c = 1; c <= clones; c++)
  {
    int numNodes = csf->numNodes;

    glm::vec4 shift = dim * 1.05f;

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
        w = float((c) / (sq * sq));
        break;
      default:
        assert(false);
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
      MatrixNode& node    = m_matrices[n + numNodes * c];
      node                = m_matrices[n];
      node.worldMatrix[3] = node.worldMatrix[3] + shift;
    }

    {
      // patch object matrix of root
      MatrixNode& node     = m_matrices[csf->rootIDX + numNodes * c];
      node.objectMatrix[3] = node.objectMatrix[3] + shift;
    }

    // clone objects
    for(int n = 0; n < numObjects; n++)
    {
      const Object& objectorig = m_objects[n];
      Object&       object     = m_objects[n + numObjects * c];

      object = objectorig;
      object.matrixIndex += c * numNodes;
      for(auto& part : object.parts)
      {
        part.matrixIndex += c * numNodes;
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

void fillMeshletTopology(NVMeshlet::PackBasicBuilder::MeshletGeometry& geometry, CadScene::MeshletTopology& topo, int useShorts)
{
  (void)useShorts;
  if(geometry.meshletDescriptors.empty())
    return;

  topo.descSize = sizeof(NVMeshlet::MeshletPackBasicDesc) * geometry.meshletDescriptors.size();
  topo.primSize = sizeof(NVMeshlet::PackBasicType) * geometry.meshletPacks.size();

  topo.descData = malloc(topo.descSize);
  topo.primData = malloc(topo.primSize);

  memcpy(topo.descData, geometry.meshletDescriptors.data(), topo.descSize);
  memcpy(topo.primData, geometry.meshletPacks.data(), topo.primSize);
}


void CadScene::buildMeshletTopology(const CSFile* csf)
{
  NVMeshlet::Stats statsGlobal;
  uint32_t         groups              = 0;
  size_t           meshActualSizeTotal = 0;

#define MESHLET_ERRORCHECK 0

  if(m_cfg.meshBuilder == MESHLET_BUILDER_PACKBASIC)
  {
    NVMeshlet::PackBasicBuilder meshletBuilder{};
    meshletBuilder.setup(m_cfg.meshVertexCount, m_cfg.meshPrimitiveCount, false);

#pragma omp parallel for
    for(int g = 0; g < csf->numGeometries; g++)
    {
      const CSFGeometry* csfgeom = csf->geometries + g;
      Geometry&          geom    = m_geometry[g];

      NVMeshlet::PackBasicBuilder::MeshletGeometry meshletGeometry;

      uint32_t               numMeshlets = 0;
      uint32_t               indexOffset = 0;
      const unsigned int*    indices     = csfgeom->indexSolid;
      const CSFGeometryPart* parts       = csfgeom->parts;
      for(size_t p = 0; p < geom.parts.size(); p++)
      {
        uint32_t numIndex              = parts[p].numIndexSolid;
        geom.parts[p].meshSolid.offset = numMeshlets;

        uint32_t processedIndices = meshletBuilder.buildMeshlets<uint32_t>(meshletGeometry, numIndex, indices + indexOffset);
        if(processedIndices != numIndex)
        {
          LOGE("warning: geometry meshlet incomplete %d\n", g)
        }

        geom.parts[p].meshSolid.count = (uint32_t)meshletGeometry.meshletDescriptors.size() - numMeshlets;
        numMeshlets                   = (uint32_t)meshletGeometry.meshletDescriptors.size();
        indexOffset += numIndex;
      }

      geom.meshlet.numMeshlets = int(meshletGeometry.meshletDescriptors.size());

      meshletBuilder.buildMeshletEarlyCulling(meshletGeometry, glm::value_ptr(m_bboxes[g].min),
                                              glm::value_ptr(m_bboxes[g].max), (const float*)csfgeom->vertex, sizeof(float) * 3);
      if(m_cfg.verbose)
      {
#if MESHLET_ERRORCHECK
        NVMeshlet::StatusCode errorcode = meshletBuilder.errorCheck<uint32_t>(meshletGeometry, 0, csfgeom->numVertices - 1,
                                                                              csfgeom->numIndexSolid, csfgeom->indexSolid);
        if(errorcode)
        {
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

      geom.meshSize        = geom.meshlet.descSize;
      geom.meshIndicesSize = geom.meshlet.primSize;

      size_t meshActualSize = geom.meshlet.descSize + geom.meshlet.primSize;

#pragma omp critical
      {
        m_meshSize += geom.meshSize + geom.meshIndicesSize;
        groups += numMeshlets;
        meshActualSizeTotal += meshActualSize;
      }
    }
  }

  LOGI("meshlet config: %d vertices, %d primitives\n", m_cfg.meshVertexCount, m_cfg.meshPrimitiveCount)

  if(m_cfg.verbose)
  {
    statsGlobal.fprint(stdout);
  }

  LOGI("meshlet total: %9d meshlets, %7zu KB (w %.2f)\n", groups, m_meshSize / 1024,
       (double(m_meshSize) / double(meshActualSizeTotal) - 1.0))
}
