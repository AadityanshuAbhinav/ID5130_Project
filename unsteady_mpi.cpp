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

struct Element
{
    std::vector<int>    ElementTag;
    std::vector<int>    NodeTags;
    std::vector<double> NodeCoords;
    std::vector<double> CentroidCoords;
    std::vector<double> Volume;

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

    // ---------------- PARSE CLI ----------------
    double lc     = (argc > 1) ? std::atof(argv[1]) : 0.15;
    int    nsteps = (argc > 2) ? std::atoi(argv[2]) : 10000;

    double delta_t = (lc * lc) / 20.0;

    // ----------------------------------------------------------------
    // MESH: rank 0 generates, broadcasts to all
    // ----------------------------------------------------------------
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

    MPI_Bcast(ElementTags_flat.data(), ElementNum,   MPI_INT,    0, MPI_COMM_WORLD);
    MPI_Bcast(NodeTags_flat.data(),    ElementNum*3, MPI_INT,    0, MPI_COMM_WORLD);
    MPI_Bcast(node_ids_int.data(),     NodeNum,      MPI_INT,    0, MPI_COMM_WORLD);
    MPI_Bcast(coords.data(),           NodeNum*3,    MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // ----------------------------------------------------------------
    // All ranks build identical full mesh struct
    // ----------------------------------------------------------------
    std::unordered_map<int,int> elementIndex;
    for(int i = 0; i < ElementNum; i++) elementIndex[ElementTags_flat[i]] = i;

    std::unordered_map<int,int> nodeIndex;
    for(int i = 0; i < NodeNum; i++) nodeIndex[node_ids_int[i]] = i;

    Element E;
    E.ElementTag.resize(ElementNum, 0);
    E.NodeTags.resize(ElementNum*3, 0);
    E.NodeCoords.resize(ElementNum*6, 0.0);
    E.CentroidCoords.resize(ElementNum*2, 0.0);
    E.Volume.resize(ElementNum, 0.0);

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
    std::vector<double> SideLengthBuffer(3, 0.0);

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
                SideLengthBuffer[j-1] = std::sqrt(Xdist*Xdist+Ydist*Ydist);
                S += SideLengthBuffer[j-1];
            }
            E.NodeTags[3*i+j]       = NodeTagBuffer[j];
            E.NodeCoords[6*i+2*j]   = coords[3*index];
            E.NodeCoords[6*i+2*j+1] = coords[3*index+1];
        }
        CentroidX /= 3.0; CentroidY /= 3.0;
        E.CentroidCoords[2*i]   = CentroidX;
        E.CentroidCoords[2*i+1] = CentroidY;

        Xdist = NodeCoordBuffer[4] - NodeCoordBuffer[0];
        Ydist = NodeCoordBuffer[5] - NodeCoordBuffer[1];
        SideLengthBuffer[2] = std::sqrt(Xdist*Xdist+Ydist*Ydist);
        S = (S + SideLengthBuffer[2]) / 2.0;
        E.Volume[i] = std::sqrt(S * (S-SideLengthBuffer[0]) * (S-SideLengthBuffer[1]) * (S-SideLengthBuffer[2]));

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

    // ---------------- NEIGHBOR DETECTION ----------------
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

    double t_setup_end = MPI_Wtime();

    // ----------------------------------------------------------------
    // BLOCK DECOMPOSITION
    // ----------------------------------------------------------------
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

    // ----------------------------------------------------------------
    // TIME STEPPING (explicit Euler)
    // ----------------------------------------------------------------
    std::vector<double> T(ElementNum, 100.0);
    std::vector<double> T_buffer(ElementNum, 100.0);

    double T_val;
    int idNeighbor;
    double PGcoeff, SGcoeff;
    std::vector<double> Af(2, 0.0), EtaCap(2, 0.0), ZhiCap(2, 0.0);
    std::vector<int> elems;

    std::ofstream file;
    if(rank == 0) file.open("Usteady_mpi.txt");

    double t_solve_start = MPI_Wtime();

    for(int t_step = 0; t_step < nsteps; t_step++)
    {
        double t_phys = delta_t * static_cast<double>(t_step);

        // each rank updates only its local slice into T_buffer
        for(int i = localStart; i < localEnd; i++)
        {
            T_val = 0.0;
            for(int j = 0; j < 3; j++)
            {
                neighbor = E.NeighborElementTags[3*i+j];

                if(neighbor != 0)
                {
                    idNeighbor = elementIndex[neighbor];
                    Af[0] = E.NeighborFaceArea[6*i+2*j];   Af[1] = E.NeighborFaceArea[6*i+2*j+1];
                    EtaCap[0] = E.NeighborEta_cap[6*i+2*j]; EtaCap[1] = E.NeighborEta_cap[6*i+2*j+1];
                    ZhiCap[0] = E.NeighborZhi_cap[6*i+2*j]; ZhiCap[1] = E.NeighborZhi_cap[6*i+2*j+1];
                    DeltaEta  = E.NeighborDeltaEta[3*i+j];  DeltaZhi  = E.NeighborDeltaZhi[3*i+j];

                    PGcoeff = (Af[0]*Af[0]+Af[1]*Af[1]) / ((Af[0]*ZhiCap[0]+Af[1]*ZhiCap[1]) * DeltaZhi);
                    SGcoeff = -(Af[0]*Af[0]+Af[1]*Af[1]) / ((Af[0]*ZhiCap[0]+Af[1]*ZhiCap[1]) * DeltaEta)
                              * (EtaCap[0]*ZhiCap[0]+EtaCap[1]*ZhiCap[1]);

                    T_val -= PGcoeff * T[i];
                    T_val += PGcoeff * T[idNeighbor];

                    nA = E.NodeTags[3*i+j];
                    nB = E.NodeTags[3*i+(j+1)%3];

                    if      (leftNodes.count(nA))   T_val -= SGcoeff * T_left;
                    else if (rightNodes.count(nA))  T_val -= SGcoeff * T_right;
                    else if (topNodes.count(nA))    T_val -= SGcoeff * T_top;
                    else if (bottomNodes.count(nA)) T_val -= SGcoeff * T_bottom;
                    else
                    {
                        elems = nodeToElements[nA];
                        for(int el = 0; el < (int)elems.size(); el++)
                            T_val -= (SGcoeff / elems.size()) * T[elems[el]];
                    }

                    if      (leftNodes.count(nB))   T_val += SGcoeff * T_left;
                    else if (rightNodes.count(nB))  T_val += SGcoeff * T_right;
                    else if (topNodes.count(nB))    T_val += SGcoeff * T_top;
                    else if (bottomNodes.count(nB)) T_val += SGcoeff * T_bottom;
                    else
                    {
                        elems = nodeToElements[nB];
                        for(int el = 0; el < (int)elems.size(); el++)
                            T_val += (SGcoeff / elems.size()) * T[elems[el]];
                    }
                }

                if (E.IsLeft[3*i+j] == 1)
                {
                    Af[0] = E.LeftFaceArea[6*i+2*j];   Af[1] = E.LeftFaceArea[6*i+2*j+1];
                    ZhiCap[0] = E.LeftZhi_cap[6*i+2*j]; ZhiCap[1] = E.LeftZhi_cap[6*i+2*j+1];
                    DeltaZhi  = E.LeftDeltaZhi[3*i+j];
                    PGcoeff = (Af[0]*Af[0]+Af[1]*Af[1]) / ((Af[0]*ZhiCap[0]+Af[1]*ZhiCap[1]) * DeltaZhi);
                    T_val += PGcoeff * T_left;  T_val -= PGcoeff * T[i];
                }
                else if (E.IsRight[3*i+j] == 1)
                {
                    Af[0] = E.RightFaceArea[6*i+2*j];   Af[1] = E.RightFaceArea[6*i+2*j+1];
                    ZhiCap[0] = E.RightZhi_cap[6*i+2*j]; ZhiCap[1] = E.RightZhi_cap[6*i+2*j+1];
                    DeltaZhi  = E.RightDeltaZhi[3*i+j];
                    PGcoeff = (Af[0]*Af[0]+Af[1]*Af[1]) / ((Af[0]*ZhiCap[0]+Af[1]*ZhiCap[1]) * DeltaZhi);
                    T_val += PGcoeff * T_right; T_val -= PGcoeff * T[i];
                }
                else if (E.IsTop[3*i+j] == 1)
                {
                    Af[0] = E.TopFaceArea[6*i+2*j];   Af[1] = E.TopFaceArea[6*i+2*j+1];
                    ZhiCap[0] = E.TopZhi_cap[6*i+2*j]; ZhiCap[1] = E.TopZhi_cap[6*i+2*j+1];
                    DeltaZhi  = E.TopDeltaZhi[3*i+j];
                    PGcoeff = (Af[0]*Af[0]+Af[1]*Af[1]) / ((Af[0]*ZhiCap[0]+Af[1]*ZhiCap[1]) * DeltaZhi);
                    T_val += PGcoeff * T_top;   T_val -= PGcoeff * T[i];
                }
                else if (E.IsBottom[3*i+j] == 1)
                {
                    Af[0] = E.BottomFaceArea[6*i+2*j];   Af[1] = E.BottomFaceArea[6*i+2*j+1];
                    ZhiCap[0] = E.BottomZhi_cap[6*i+2*j]; ZhiCap[1] = E.BottomZhi_cap[6*i+2*j+1];
                    DeltaZhi  = E.BottomDeltaZhi[3*i+j];
                    PGcoeff = (Af[0]*Af[0]+Af[1]*Af[1]) / ((Af[0]*ZhiCap[0]+Af[1]*ZhiCap[1]) * DeltaZhi);
                    T_val += PGcoeff * T_bottom; T_val -= PGcoeff * T[i];
                }
            }
            T_val *= (delta_t / E.Volume[i]);
            T_val += T[i];
            T_buffer[i] = T_val;
        }

        // gather local slices of T_buffer into T (T_buffer != T — no aliasing)
        MPI_Allgatherv(T_buffer.data() + localStart, localSize, MPI_DOUBLE,
                       T.data(), recvcounts.data(), displs.data(),
                       MPI_DOUBLE, MPI_COMM_WORLD);

        // rank 0 writes every 10 steps
        if(rank == 0 && t_step % 10 == 0)
        {
            for(int i = 0; i < ElementNum; i++)
                file << t_phys << " " << E.CentroidCoords[2*i] << " "
                     << E.CentroidCoords[2*i+1] << " " << T[i] << "\n";
        }
    }

    if(rank == 0) file.close();

    double t_solve_end = MPI_Wtime();

    if(rank == 0)
    {
        double setup_time  = t_setup_end  - t_setup_start;
        double solver_time = t_solve_end  - t_solve_start;
        double total_time  = t_solve_end  - t_setup_start;

        std::cout << "PERF unsteady mpi "
                  << size        << " "
                  << lc          << " "
                  << ElementNum  << " "
                  << setup_time  << " "
                  << solver_time << " "
                  << total_time  << " "
                  << nsteps      << "\n";
    }

    MPI_Finalize();
    return 0;
}
