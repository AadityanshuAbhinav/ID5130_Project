#include <mpi.h>
#include <gmsh.h>
#include <vector>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <cmath>
#include <fstream>

const double tol = 1e-8;
const double PI  = 3.14159265358979;

const double T_left   = 50.0;
const double T_right  = 200.0;
const double T_top    = 100.0;
const double T_bottom = 300.0;

double T_analytical(double x, double y, int n)
{
    double T = 0;
    for (int i = 1; i <= n; i++)
    {
        if (i % 2 != 0)
        {
            double I = static_cast<double>(i);
            T += T_left   * (4.0/(I*PI)) * std::sin(I*PI*y) * (std::sinh(I*PI*(1.0-x)) / std::sinh(I*PI));
            T += T_bottom * (4.0/(I*PI)) * std::sin(I*PI*x) * (std::sinh(I*PI*(1.0-y)) / std::sinh(I*PI));
            T += T_right  * (4.0/(I*PI)) * std::sin(I*PI*y) * (std::sinh(I*PI*x)       / std::sinh(I*PI));
            T += T_top    * (4.0/(I*PI)) * std::sin(I*PI*x) * (std::sinh(I*PI*y)       / std::sinh(I*PI));
        }
    }
    return T;
}

struct Element
{
    std::vector<int>    ElementTag;
    std::vector<int>    NodeTags;
    std::vector<double> NodeCoords;
    std::vector<double> CentroidCoords;

    std::vector<int>    NeighborElementTags;
    std::vector<double> NeighborEta_cap;
    std::vector<double> NeighborDeltaEta;
    std::vector<double> NeighborZhi_cap;
    std::vector<double> NeighborDeltaZhi;
    std::vector<double> NeighborFaceArea;

    std::vector<int>    IsLeft;
    std::vector<double> LeftEta_cap;
    std::vector<double> LeftDeltaEta;
    std::vector<double> LeftZhi_cap;
    std::vector<double> LeftDeltaZhi;
    std::vector<double> LeftFaceArea;

    std::vector<int>    IsRight;
    std::vector<double> RightEta_cap;
    std::vector<double> RightDeltaEta;
    std::vector<double> RightZhi_cap;
    std::vector<double> RightDeltaZhi;
    std::vector<double> RightFaceArea;

    std::vector<int>    IsTop;
    std::vector<double> TopEta_cap;
    std::vector<double> TopDeltaEta;
    std::vector<double> TopZhi_cap;
    std::vector<double> TopDeltaZhi;
    std::vector<double> TopFaceArea;

    std::vector<int>    IsBottom;
    std::vector<double> BottomEta_cap;
    std::vector<double> BottomDeltaEta;
    std::vector<double> BottomZhi_cap;
    std::vector<double> BottomDeltaZhi;
    std::vector<double> BottomFaceArea;
};

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double lc    = (argc > 1) ? std::atof(argv[1]) : 0.15;
    double omega = (argc > 2) ? std::atof(argv[2]) : 1.5;

    int ElementNum = 0, NodeNum = 0;
    std::vector<int>    ElementTags_flat;
    std::vector<int>    NodeTags_flat;
    std::vector<int>    node_ids_int;
    std::vector<double> coords;

    double t_setup_start = MPI_Wtime();

    if(rank == 0)
    {
        gmsh::initialize();
        gmsh::option::setNumber("General.Verbosity", 0);
        gmsh::model::add("square");

        gmsh::model::geo::addPoint(0,0,0,lc,1);
        gmsh::model::geo::addPoint(1,0,0,lc,2);
        gmsh::model::geo::addPoint(1,1,0,lc,3);
        gmsh::model::geo::addPoint(0,1,0,lc,4);

        gmsh::model::geo::addLine(1,2,1);
        gmsh::model::geo::addLine(2,3,2);
        gmsh::model::geo::addLine(3,4,3);
        gmsh::model::geo::addLine(4,1,4);

        gmsh::model::geo::addCurveLoop({1,2,3,4},1);
        gmsh::model::geo::addPlaneSurface({1},1);
        gmsh::model::geo::synchronize();
        gmsh::model::mesh::generate(2);
        gmsh::write("mesh.msh");

        std::vector<int> types;
        std::vector<std::vector<std::size_t>> elemTagsVec, nodeTagsVec;
        gmsh::model::mesh::getElements(types, elemTagsVec, nodeTagsVec);

        int tri_idx = -1;
        for(int i = 0; i < (int)types.size(); i++)
            if(types[i] == 2){ tri_idx = i; break; }

        ElementNum = (int)elemTagsVec[tri_idx].size();
        ElementTags_flat.resize(ElementNum);
        NodeTags_flat.resize(ElementNum * 3);
        for(int i = 0; i < ElementNum; i++) ElementTags_flat[i] = (int)elemTagsVec[tri_idx][i];
        for(int i = 0; i < ElementNum*3; i++) NodeTags_flat[i]  = (int)nodeTagsVec[tri_idx][i];

        std::vector<std::size_t> node_ids_sz;
        std::vector<double> param;
        gmsh::model::mesh::getNodes(node_ids_sz, coords, param);
        gmsh::finalize();

        NodeNum = (int)node_ids_sz.size();
        node_ids_int.resize(NodeNum);
        for(int i = 0; i < NodeNum; i++) node_ids_int[i] = (int)node_ids_sz[i];

        std::cout << "Triangles: " << ElementNum << "  Nodes: " << NodeNum << "\n";
    }

    MPI_Bcast(&ElementNum, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&NodeNum,    1, MPI_INT, 0, MPI_COMM_WORLD);

    if(rank != 0)
    {
        ElementTags_flat.resize(ElementNum);
        NodeTags_flat.resize(ElementNum * 3);
        node_ids_int.resize(NodeNum);
        coords.resize(NodeNum * 3);
    }

    MPI_Bcast(ElementTags_flat.data(), ElementNum,     MPI_INT,    0, MPI_COMM_WORLD);
    MPI_Bcast(NodeTags_flat.data(),    ElementNum*3,   MPI_INT,    0, MPI_COMM_WORLD);
    MPI_Bcast(node_ids_int.data(),     NodeNum,        MPI_INT,    0, MPI_COMM_WORLD);
    MPI_Bcast(coords.data(),           NodeNum*3,      MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::unordered_map<int,int> elementIndex;
    for(int i = 0; i < ElementNum; i++) elementIndex[ElementTags_flat[i]] = i;

    std::unordered_map<int,int> nodeIndex;
    for(int i = 0; i < NodeNum; i++) nodeIndex[node_ids_int[i]] = i;

    Element E;
    E.ElementTag.resize(ElementNum, 0);
    E.NodeTags.resize(ElementNum*3, 0);
    E.NodeCoords.resize(ElementNum*6, 0.0);
    E.CentroidCoords.resize(ElementNum*2, 0.0);

    E.NeighborElementTags.resize(ElementNum*3, 0);
    E.NeighborEta_cap.resize(ElementNum*6, 0.0);
    E.NeighborDeltaEta.resize(ElementNum*3, 0.0);
    E.NeighborZhi_cap.resize(ElementNum*6, 0.0);
    E.NeighborDeltaZhi.resize(ElementNum*3, 0.0);
    E.NeighborFaceArea.resize(ElementNum*6, 0.0);

    E.IsLeft.resize(ElementNum*3, 0);
    E.LeftEta_cap.resize(ElementNum*6, 0.0);
    E.LeftDeltaEta.resize(ElementNum*3, 0.0);
    E.LeftZhi_cap.resize(ElementNum*6, 0.0);
    E.LeftDeltaZhi.resize(ElementNum*3, 0.0);
    E.LeftFaceArea.resize(ElementNum*6, 0.0);

    E.IsRight.resize(ElementNum*3, 0);
    E.RightEta_cap.resize(ElementNum*6, 0.0);
    E.RightDeltaEta.resize(ElementNum*3, 0.0);
    E.RightZhi_cap.resize(ElementNum*6, 0.0);
    E.RightDeltaZhi.resize(ElementNum*3, 0.0);
    E.RightFaceArea.resize(ElementNum*6, 0.0);

    E.IsTop.resize(ElementNum*3, 0);
    E.TopEta_cap.resize(ElementNum*6, 0.0);
    E.TopDeltaEta.resize(ElementNum*3, 0.0);
    E.TopZhi_cap.resize(ElementNum*6, 0.0);
    E.TopDeltaZhi.resize(ElementNum*3, 0.0);
    E.TopFaceArea.resize(ElementNum*6, 0.0);

    E.IsBottom.resize(ElementNum*3, 0);
    E.BottomEta_cap.resize(ElementNum*6, 0.0);
    E.BottomDeltaEta.resize(ElementNum*3, 0.0);
    E.BottomZhi_cap.resize(ElementNum*6, 0.0);
    E.BottomDeltaZhi.resize(ElementNum*3, 0.0);
    E.BottomFaceArea.resize(ElementNum*6, 0.0);

    std::unordered_set<int> leftNodes, rightNodes, topNodes, bottomNodes;
    std::unordered_map<int, std::vector<int>> nodeToElements;
    for(int i = 0; i < ElementNum; i++)
        for(int j = 0; j < 3; j++)
            nodeToElements[NodeTags_flat[3*i+j]].push_back(i);

    std::vector<int>    NodeTagBuffer(3, 0);
    std::vector<double> NodeCoordBuffer(6, 0.0);
    int index;
    double CentroidX, CentroidY, S, Xdist, Ydist;
    double CentroidSideX, CentroidSideY;
    int nA, nB, idxA, idxB;
    double xA, xB, yA, yB, etaX, etaY, zhiX, zhiY, val;

    for(int i = 0; i < ElementNum; i++)
    {
        E.ElementTag[i] = ElementTags_flat[i];
        CentroidX = CentroidY = S = 0.0;
        for(int j = 0; j < 3; j++)
        {
            NodeTagBuffer[j] = NodeTags_flat[3*i+j];
            index = nodeIndex[NodeTagBuffer[j]];
            CentroidX += coords[3*index];
            CentroidY += coords[3*index+1];
            NodeCoordBuffer[2*j]   = coords[3*index];
            NodeCoordBuffer[2*j+1] = coords[3*index+1];
            if(j >= 1)
            {
                Xdist = NodeCoordBuffer[2*j]   - NodeCoordBuffer[2*(j-1)];
                Ydist = NodeCoordBuffer[2*j+1] - NodeCoordBuffer[2*(j-1)+1];
                S += std::sqrt(Xdist*Xdist+Ydist*Ydist);
            }
            E.NodeTags[3*i+j]       = NodeTagBuffer[j];
            E.NodeCoords[6*i+2*j]   = coords[3*index];
            E.NodeCoords[6*i+2*j+1] = coords[3*index+1];
        }
        CentroidX /= 3.0; CentroidY /= 3.0;
        E.CentroidCoords[2*i]   = CentroidX;
        E.CentroidCoords[2*i+1] = CentroidY;

        for(int j = 0; j < 3; j++)
        {
            nA = E.NodeTags[3*i+j]; nB = E.NodeTags[3*i+(j+1)%3];
            idxA = nodeIndex[nA];   idxB = nodeIndex[nB];
            xA = coords[3*idxA]; yA = coords[3*idxA+1];
            xB = coords[3*idxB]; yB = coords[3*idxB+1];

            if(std::abs(xA) < tol && std::abs(xB) < tol)
            {
                E.IsLeft[3*i+j] = 1;
                CentroidSideX = (xA+xB)/2.0; CentroidSideY = (yA+yB)/2.0;
                etaX = xB-xA; etaY = yB-yA;
                E.LeftFaceArea[6*i+2*j] = etaY; E.LeftFaceArea[6*i+2*j+1] = -etaX;
                zhiX = CentroidSideX-CentroidX; zhiY = CentroidSideY-CentroidY;
                val = std::sqrt(etaX*etaX+etaY*etaY);
                E.LeftDeltaEta[3*i+j] = val;
                E.LeftEta_cap[6*i+2*j] = etaX/val; E.LeftEta_cap[6*i+2*j+1] = etaY/val;
                val = std::sqrt(zhiX*zhiX+zhiY*zhiY);
                E.LeftDeltaZhi[3*i+j] = val;
                E.LeftZhi_cap[6*i+2*j] = zhiX/val; E.LeftZhi_cap[6*i+2*j+1] = zhiY/val;
                leftNodes.insert(nA); leftNodes.insert(nB);
            }
            if(std::abs(xA-1.0) < tol && std::abs(xB-1.0) < tol)
            {
                E.IsRight[3*i+j] = 1;
                CentroidSideX = (xA+xB)/2.0; CentroidSideY = (yA+yB)/2.0;
                etaX = xB-xA; etaY = yB-yA;
                E.RightFaceArea[6*i+2*j] = etaY; E.RightFaceArea[6*i+2*j+1] = -etaX;
                zhiX = CentroidSideX-CentroidX; zhiY = CentroidSideY-CentroidY;
                val = std::sqrt(etaX*etaX+etaY*etaY);
                E.RightDeltaEta[3*i+j] = val;
                E.RightEta_cap[6*i+2*j] = etaX/val; E.RightEta_cap[6*i+2*j+1] = etaY/val;
                val = std::sqrt(zhiX*zhiX+zhiY*zhiY);
                E.RightDeltaZhi[3*i+j] = val;
                E.RightZhi_cap[6*i+2*j] = zhiX/val; E.RightZhi_cap[6*i+2*j+1] = zhiY/val;
                rightNodes.insert(nA); rightNodes.insert(nB);
            }
            if(std::abs(yA) < tol && std::abs(yB) < tol)
            {
                E.IsBottom[3*i+j] = 1;
                CentroidSideX = (xA+xB)/2.0; CentroidSideY = (yA+yB)/2.0;
                etaX = xB-xA; etaY = yB-yA;
                E.BottomFaceArea[6*i+2*j] = etaY; E.BottomFaceArea[6*i+2*j+1] = -etaX;
                zhiX = CentroidSideX-CentroidX; zhiY = CentroidSideY-CentroidY;
                val = std::sqrt(etaX*etaX+etaY*etaY);
                E.BottomDeltaEta[3*i+j] = val;
                E.BottomEta_cap[6*i+2*j] = etaX/val; E.BottomEta_cap[6*i+2*j+1] = etaY/val;
                val = std::sqrt(zhiX*zhiX+zhiY*zhiY);
                E.BottomDeltaZhi[3*i+j] = val;
                E.BottomZhi_cap[6*i+2*j] = zhiX/val; E.BottomZhi_cap[6*i+2*j+1] = zhiY/val;
                bottomNodes.insert(nA); bottomNodes.insert(nB);
            }
            if(std::abs(yA-1.0) < tol && std::abs(yB-1.0) < tol)
            {
                E.IsTop[3*i+j] = 1;
                CentroidSideX = (xA+xB)/2.0; CentroidSideY = (yA+yB)/2.0;
                etaX = xB-xA; etaY = yB-yA;
                E.TopFaceArea[6*i+2*j] = etaY; E.TopFaceArea[6*i+2*j+1] = -etaX;
                zhiX = CentroidSideX-CentroidX; zhiY = CentroidSideY-CentroidY;
                val = std::sqrt(etaX*etaX+etaY*etaY);
                E.TopDeltaEta[3*i+j] = val;
                E.TopEta_cap[6*i+2*j] = etaX/val; E.TopEta_cap[6*i+2*j+1] = etaY/val;
                val = std::sqrt(zhiX*zhiX+zhiY*zhiY);
                E.TopDeltaZhi[3*i+j] = val;
                E.TopZhi_cap[6*i+2*j] = zhiX/val; E.TopZhi_cap[6*i+2*j+1] = zhiY/val;
                topNodes.insert(nA); topNodes.insert(nB);
            }
        }
    }

    int n1, n2, n3, a, b, otherElem, otherEdge;
    std::map<std::pair<int,int>, std::pair<int,int>> edgeMap;
    for(int i = 0; i < ElementNum; i++)
    {
        n1 = E.NodeTags[3*i]; n2 = E.NodeTags[3*i+1]; n3 = E.NodeTags[3*i+2];
        int edges[3][2] = {{n1,n2},{n2,n3},{n3,n1}};
        for(int e = 0; e < 3; e++)
        {
            a = edges[e][0]; b = edges[e][1];
            if(a > b) std::swap(a,b);
            auto key = std::make_pair(a,b);
            auto it  = edgeMap.find(key);
            if(it == edgeMap.end()) { edgeMap[key] = {i,e}; }
            else
            {
                otherElem = it->second.first; otherEdge = it->second.second;
                E.NeighborElementTags[3*i+e]               = E.ElementTag[otherElem];
                E.NeighborElementTags[3*otherElem+otherEdge] = E.ElementTag[i];
            }
        }
    }

    double cx_i, cy_i;
    int neighbor;
    double DeltaEta, DeltaZhi;

    for(int i = 0; i < ElementNum; i++)
    {
        cx_i = E.CentroidCoords[2*i]; cy_i = E.CentroidCoords[2*i+1];
        for(int j = 0; j < 3; j++)
        {
            neighbor = E.NeighborElementTags[3*i+j];
            if(neighbor == 0) continue;

            nA = E.NodeTags[3*i+j]; nB = E.NodeTags[3*i+(j+1)%3];
            idxA = nodeIndex[nA];   idxB = nodeIndex[nB];
            xA = coords[3*idxA]; yA = coords[3*idxA+1];
            xB = coords[3*idxB]; yB = coords[3*idxB+1];

            etaX = xB-xA; etaY = yB-yA;
            DeltaEta = std::sqrt(etaX*etaX+etaY*etaY);
            etaX /= DeltaEta; etaY /= DeltaEta;
            E.NeighborEta_cap[6*i+2*j]   = etaX;
            E.NeighborEta_cap[6*i+2*j+1] = etaY;
            E.NeighborDeltaEta[3*i+j]    = DeltaEta;
            E.NeighborFaceArea[6*i+2*j]   =  etaY * DeltaEta;
            E.NeighborFaceArea[6*i+2*j+1] = -etaX * DeltaEta;

            int idx_nb = elementIndex[neighbor];
            double cx_nb = E.CentroidCoords[2*idx_nb];
            double cy_nb = E.CentroidCoords[2*idx_nb+1];
            double zhiX_ = cx_nb-cx_i, zhiY_ = cy_nb-cy_i;
            DeltaZhi = std::sqrt(zhiX_*zhiX_+zhiY_*zhiY_);
            E.NeighborZhi_cap[6*i+2*j]   = zhiX_/DeltaZhi;
            E.NeighborZhi_cap[6*i+2*j+1] = zhiY_/DeltaZhi;
            E.NeighborDeltaZhi[3*i+j]    = DeltaZhi;
        }
    }

    int localStart = rank * ElementNum / size;
    int localEnd   = (rank + 1) * ElementNum / size;
    int localSize  = localEnd - localStart;

    std::vector<int> recvcounts(size), displs(size);
    for(int r = 0; r < size; r++)
    {
        int rs = r * ElementNum / size;
        int re = (r+1) * ElementNum / size;
        recvcounts[r] = re - rs;
        displs[r]     = rs;
    }

    std::vector<double> A_local(localSize * ElementNum, 0.0);
    std::vector<double> B_local(localSize, 0.0);

    int idNeighbor;
    double PGcoeff, SGcoeff;
    std::vector<double> Af(2), EtaCap(2), ZhiCap(2);
    std::vector<int> elems;
    int n_analytical = 100;

    std::vector<double> T_exact(ElementNum, 0.0);
    for(int i = 0; i < ElementNum; i++)
        T_exact[i] = T_analytical(E.CentroidCoords[2*i], E.CentroidCoords[2*i+1], n_analytical);

    for(int i = localStart; i < localEnd; i++)
    {
        int li = i - localStart;

        for(int j = 0; j < 3; j++)
        {
            neighbor = E.NeighborElementTags[3*i+j];

            if(neighbor != 0)
            {
                idNeighbor = elementIndex[neighbor];
                Af[0] = E.NeighborFaceArea[6*i+2*j];   Af[1] = E.NeighborFaceArea[6*i+2*j+1];
                EtaCap[0] = E.NeighborEta_cap[6*i+2*j]; EtaCap[1] = E.NeighborEta_cap[6*i+2*j+1];
                ZhiCap[0] = E.NeighborZhi_cap[6*i+2*j]; ZhiCap[1] = E.NeighborZhi_cap[6*i+2*j+1];
                DeltaEta = E.NeighborDeltaEta[3*i+j];   DeltaZhi = E.NeighborDeltaZhi[3*i+j];

                PGcoeff = (Af[0]*Af[0]+Af[1]*Af[1]) / ((Af[0]*ZhiCap[0]+Af[1]*ZhiCap[1]) * DeltaZhi);
                SGcoeff = -(Af[0]*Af[0]+Af[1]*Af[1]) / ((Af[0]*ZhiCap[0]+Af[1]*ZhiCap[1]) * DeltaEta)
                          * (EtaCap[0]*ZhiCap[0]+EtaCap[1]*ZhiCap[1]);

                A_local[li*ElementNum+i]          -= PGcoeff;
                A_local[li*ElementNum+idNeighbor] += PGcoeff;

                nA = E.NodeTags[3*i+j]; nB = E.NodeTags[3*i+(j+1)%3];

                if      (leftNodes.count(nA))   B_local[li] += SGcoeff * T_left;
                else if (rightNodes.count(nA))  B_local[li] += SGcoeff * T_right;
                else if (topNodes.count(nA))    B_local[li] += SGcoeff * T_top;
                else if (bottomNodes.count(nA)) B_local[li] += SGcoeff * T_bottom;
                else
                {
                    elems = nodeToElements[nA];
                    for(int el = 0; el < (int)elems.size(); el++)
                        A_local[li*ElementNum+elems[el]] -= SGcoeff / elems.size();
                }

                if      (leftNodes.count(nB))   B_local[li] -= SGcoeff * T_left;
                else if (rightNodes.count(nB))  B_local[li] -= SGcoeff * T_right;
                else if (topNodes.count(nB))    B_local[li] -= SGcoeff * T_top;
                else if (bottomNodes.count(nB)) B_local[li] -= SGcoeff * T_bottom;
                else
                {
                    elems = nodeToElements[nB];
                    for(int el = 0; el < (int)elems.size(); el++)
                        A_local[li*ElementNum+elems[el]] += SGcoeff / elems.size();
                }
            }

            auto applyBC_local = [&](std::vector<double>& FaceArea,
                                     std::vector<double>& Zhi_cap,
                                     std::vector<double>& DeltaZhi_v,
                                     double T_bc)
            {
                Af[0] = FaceArea[6*i+2*j]; Af[1] = FaceArea[6*i+2*j+1];
                ZhiCap[0] = Zhi_cap[6*i+2*j]; ZhiCap[1] = Zhi_cap[6*i+2*j+1];
                double dz = DeltaZhi_v[3*i+j];
                double pg = (Af[0]*Af[0]+Af[1]*Af[1]) / ((Af[0]*ZhiCap[0]+Af[1]*ZhiCap[1]) * dz);
                B_local[li] -= pg * T_bc;
                A_local[li*ElementNum+i] -= pg;
            };

            if      (E.IsLeft[3*i+j]   == 1) applyBC_local(E.LeftFaceArea,   E.LeftZhi_cap,   E.LeftDeltaZhi,   T_left);
            else if (E.IsRight[3*i+j]  == 1) applyBC_local(E.RightFaceArea,  E.RightZhi_cap,  E.RightDeltaZhi,  T_right);
            else if (E.IsTop[3*i+j]    == 1) applyBC_local(E.TopFaceArea,    E.TopZhi_cap,    E.TopDeltaZhi,    T_top);
            else if (E.IsBottom[3*i+j] == 1) applyBC_local(E.BottomFaceArea, E.BottomZhi_cap, E.BottomDeltaZhi, T_bottom);
        }
    }

    std::vector<std::vector<int>> colorGroups;
    {
        std::vector<std::unordered_set<int>> adjSet(ElementNum);
        for (auto& kv : nodeToElements) {
            const auto& ev = kv.second;
            for (int a = 0; a < (int)ev.size(); a++)
                for (int b = a+1; b < (int)ev.size(); b++) {
                    adjSet[ev[a]].insert(ev[b]);
                    adjSet[ev[b]].insert(ev[a]);
                }
        }
        std::vector<int> color(ElementNum, -1);
        for (int i = 0; i < ElementNum; i++) {
            std::unordered_set<int> used;
            for (int nb : adjSet[i])
                if (color[nb] >= 0) used.insert(color[nb]);
            int c = 0;
            while (used.count(c)) c++;
            color[i] = c;
            if (c >= (int)colorGroups.size()) colorGroups.resize(c + 1);
            colorGroups[c].push_back(i);
        }
    }

    double t_setup_end = MPI_Wtime();

    std::vector<double> T(ElementNum, 100.0);

    int maxIter = 10000;
    double tolGS = 1e-10;
    int convIter = maxIter;

    if (rank == 0)
        std::cout << "omega = " << omega << "  colors = " << colorGroups.size() << "\n";

    double t_solve_start = MPI_Wtime();

    for(int iter = 0; iter < maxIter; iter++)
    {
        double localMax = 0.0;

        for (const auto& group : colorGroups)
        {
            for (int i : group)
            {
                if (i < localStart || i >= localEnd) continue;
                int li = i - localStart;
                double sigma = 0.0;
                for(int j = 0; j < ElementNum; j++)
                    if(j != i) sigma += A_local[li*ElementNum+j] * T[j];
                double T_gs  = (B_local[li] - sigma) / A_local[li*ElementNum+i];
                double T_new = (1.0 - omega) * T[i] + omega * T_gs;
                localMax = std::max(localMax, std::abs(T_new - T[i]));
                T[i] = T_new;
            }
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DOUBLE,
                           T.data(), recvcounts.data(), displs.data(),
                           MPI_DOUBLE, MPI_COMM_WORLD);
        }

        double globalMax = 0.0;
        MPI_Allreduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        if(globalMax < tolGS){ convIter = iter; break; }
        if(iter == maxIter-1 && rank == 0) std::cout << "Did not converge\n";
    }

    double t_solve_end = MPI_Wtime();

    if(rank == 0)
    {
        double maxDiff = 0.0;
        for(int i = 0; i < ElementNum; i++)
            maxDiff = std::max(maxDiff, std::abs(T[i] - T_exact[i]));
        std::cout << "Converged in " << convIter << " iterations  Max error: " << maxDiff << "\n";

        std::ofstream outf("steady_result_mpi.txt");
        outf << "x y T T_exact\n";
        for(int i = 0; i < ElementNum; i++)
            outf << E.CentroidCoords[2*i] << " " << E.CentroidCoords[2*i+1]
                 << " " << T[i] << " " << T_exact[i] << "\n";
        outf.close();

        double setup_time  = t_setup_end  - t_setup_start;
        double solver_time = t_solve_end  - t_solve_start;
        double total_time  = t_solve_end  - t_setup_start;

        std::cout << "PERF steady mpi "
                  << size        << " "
                  << lc          << " "
                  << ElementNum  << " "
                  << setup_time  << " "
                  << solver_time << " "
                  << total_time  << " "
                  << convIter    << "\n";
    }

    MPI_Finalize();
    return 0;
}
