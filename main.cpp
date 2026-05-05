#include <gmsh.h>
#include <vector>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <cmath>

const double tol = 1e-8;
const double lc = 0.15;

const double T_left = 50.0;
const double T_right = 200.0;
const double T_top = 100.0;
const double T_bottom = 300.0;

const double PI = 3.14159265;

double T_analytical(double x, double y, int n)
{
    double T = 0;
    for (int i = 1; i <= n; i++)
    {
        if (i % 2 != 0)
        {
            double I = static_cast<double>(i);
            T += T_left * (4.0 / (I * PI)) * std::sin(I * PI * y) * (std::sinh(I * PI * (1.0 - x)) / std::sinh(I * PI));
            T += T_bottom * (4.0 / (I * PI)) * std::sin(I * PI * x) * (std::sinh(I * PI * (1.0 - y)) / std::sinh(I * PI));
            T += T_right * (4.0 / (I * PI)) * std::sin(I * PI * y) * (std::sinh(I * PI * x) / std::sinh(I * PI));
            T += T_top * (4.0 / (I * PI)) * std::sin(I * PI * x) * (std::sinh(I * PI * y) / std::sinh(I * PI));
        }
    }
    return T;
}

struct Element
{
    // ---------------- ELEMENT DATA ----------------
    std::vector<int> ElementTag; // [ElementTag1, ElementTag 2, ....]
    std::vector<int> NodeTags; // [n11, n12, n13, n21, n22, n23, ....]
    std::vector<double> NodeCoords; // [x1, y1, x2, y2, x3, y3, ....]
    std::vector<double> CentroidCoords; // [x, y, x, y, ....]

    // ---------------- NEIGHBORING ELEMENTS DATA ----------------
    std::vector<int> NeighborElementTags; // [Neighbor11, Neighbor12, Neighbor13, Neighbor21, Neighbor22, Neighbor23, ....]
    // If no neighbor, then Tag set to 0
    std::vector<double> NeighborEta_cap; // [Eta1x, Eta1y, Eta2x, Eta2y, ....]
    std::vector<double> NeighborDeltaEta; // [DeltaEta11, DeltaEta12, DeltaEta13, ....]
    std::vector<double> NeighborZhi_cap; //similar to above
    std::vector<double> NeighborDeltaZhi; //similar to above
    std::vector<double> NeighborFaceArea; // same format as Eta_cap, Zhi_cap

    // ---------------- LEFT DATA ----------------
    std::vector<int> IsLeft; // binary val, 1 if the boundary is left, 0 if not left
    std::vector<double> LeftEta_cap;
    std::vector<double> LeftDeltaEta;
    std::vector<double> LeftZhi_cap;
    std::vector<double> LeftDeltaZhi;
    std::vector<double> LeftFaceArea;

    // ---------------- RIGHT DATA ----------------
    std::vector<int> IsRight;
    std::vector<double> RightEta_cap;
    std::vector<double> RightDeltaEta;
    std::vector<double> RightZhi_cap;
    std::vector<double> RightDeltaZhi;
    std::vector<double> RightFaceArea;

    // ---------------- TOP DATA ----------------
    std::vector<int> IsTop;
    std::vector<double> TopEta_cap;
    std::vector<double> TopDeltaEta;
    std::vector<double> TopZhi_cap;
    std::vector<double> TopDeltaZhi;
    std::vector<double> TopFaceArea;

    // ---------------- TOP DATA ----------------
    std::vector<int> IsBottom;
    std::vector<double> BottomEta_cap;
    std::vector<double> BottomDeltaEta;
    std::vector<double> BottomZhi_cap;
    std::vector<double> BottomDeltaZhi;
    std::vector<double> BottomFaceArea;
};

int main(){

    // ---------------- CREATE GMSH FILE ----------------
    gmsh::initialize();
    gmsh::model::add("square");

    // Geometry
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

    // ---------------- DONE CREATING GMSH FILE ----------------

    // ---------------- GET ELEMENTS ----------------

    std::vector<int> types;
    std::vector<std::vector<std::size_t>> elementTags;
    std::vector<std::vector<std::size_t>> nodeTags;

    gmsh::model::mesh::getElements(types, elementTags, nodeTags);

    int tri_idx = -1;
    for(int i = 0; i < types.size(); i++){
        if(types[i] == 2){
            tri_idx = i;
            break;
        }
    }

    const auto &elems_sz = elementTags[tri_idx];
    const auto &nodes_sz = nodeTags[tri_idx];

    // Convert to int (fixed size)
    std::vector<int> ElementTags(elems_sz.size());
    std::vector<int> NodeTags(nodes_sz.size());

    for(int i = 0; i < elems_sz.size(); i++){
        ElementTags[i] = static_cast<int>(elems_sz[i]);
    }

    // ---------------- BUILD ELEMENT LOOKUP TABLE ----------------
    // this tells us the index corresponding to each Element Tag. The same id as per what we will fill later in the struct
    std::unordered_map<int, int> elementIndex;

    for(int i = 0; i < ElementTags.size(); i++){
        elementIndex[ElementTags[i]] = i;
    }

    for(int i = 0; i < nodes_sz.size(); i++){
        NodeTags[i] = static_cast<int>(nodes_sz[i]);
    }

    // ---------------- GET NODES (RAW) ----------------

    std::vector<std::size_t> node_ids;
    std::vector<double> coords;
    std::vector<double> param;

    gmsh::model::mesh::getNodes(node_ids, coords, param);

    // ---------------- BUILD LOOKUP TABLE FOR NODES ----------------
    // a similar thing with the nodes
    std::unordered_map<int, int> nodeIndex;

    for(int i = 0; i < node_ids.size(); i++){
        nodeIndex[static_cast<int>(node_ids[i])] = i;
    }

    gmsh::finalize();

    // ---------------- DEBUG ----------------

    std::cout << "Triangles: " << ElementTags.size() << "\n";
    std::cout << "Total nodes: " << node_ids.size() << "\n";
    std::cout << "Coords size: " << coords.size() << "\n";

    int ElementNum = ElementTags.size(); //number of elements that are triangular
    int NodeNum = node_ids.size(); //number of nodes

    // ---------------- RESIZING STRUCT ----------------

    Element E;

    E.ElementTag.resize(ElementNum, 0);
    E.NodeTags.resize(ElementNum * 3, 0);
    E.NodeCoords.resize(ElementNum * 6, 0.0);
    E.CentroidCoords.resize(ElementNum * 2, 0.0);

    E.NeighborElementTags.resize(ElementNum * 3, 0);
    E.NeighborEta_cap.resize(ElementNum * 6, 0.0);
    E.NeighborDeltaEta.resize(ElementNum * 3, 0.0);
    E.NeighborZhi_cap.resize(ElementNum * 6, 0.0);
    E.NeighborDeltaZhi.resize(ElementNum * 3, 0.0);
    E.NeighborFaceArea.resize(ElementNum * 3, 0.0);

    E.IsLeft.resize(ElementNum * 3, 0);
    E.LeftEta_cap.resize(ElementNum * 6, 0.0);
    E.LeftDeltaEta.resize(ElementNum * 3, 0.0);
    E.LeftZhi_cap.resize(ElementNum * 6, 0.0);
    E.LeftDeltaZhi.resize(ElementNum * 3, 0.0);
    E.LeftFaceArea.resize(ElementNum * 6, 0.0);

    E.IsRight.resize(ElementNum * 3, 0);
    E.RightEta_cap.resize(ElementNum * 6, 0.0);
    E.RightDeltaEta.resize(ElementNum * 3, 0.0);
    E.RightZhi_cap.resize(ElementNum * 6, 0.0);
    E.RightDeltaZhi.resize(ElementNum * 3, 0.0);
    E.RightFaceArea.resize(ElementNum * 6, 0.0);

    E.IsTop.resize(ElementNum * 3, 0);
    E.TopEta_cap.resize(ElementNum * 6, 0.0);
    E.TopDeltaEta.resize(ElementNum * 3, 0.0);
    E.TopZhi_cap.resize(ElementNum * 6, 0.0);
    E.TopDeltaZhi.resize(ElementNum * 3, 0.0);
    E.TopFaceArea.resize(ElementNum * 6, 0.0);

    E.IsBottom.resize(ElementNum * 3, 0);
    E.BottomEta_cap.resize(ElementNum * 6, 0.0);
    E.BottomDeltaEta.resize(ElementNum * 3, 0.0);
    E.BottomZhi_cap.resize(ElementNum * 6, 0.0);
    E.BottomDeltaZhi.resize(ElementNum * 3, 0.0);
    E.BottomFaceArea.resize(ElementNum * 6, 0.0);

    // ---------------- NODES ON BOUNDARIES ----------------
    std::unordered_set<int> leftNodes;
    std::unordered_set<int> rightNodes;
    std::unordered_set<int> topNodes;
    std::unordered_set<int> bottomNodes;

    // ---------------- NODE TO ELEMENT LOOKUP ----------------
    std::unordered_map<int, std::vector<int>> nodeToElements;

    //Buffer variables
    std::vector<int> NodeTagBuffer(3, 0);
    std::vector<double> NodeCoordBuffer(6, 0.0);
    int index;

    double CentroidX;
    double CentroidY;

    std::vector<double> SideLengthBuffer(3, 0.0);
    double S;
    double Xdist, Ydist;

    double CentroidSideX;
    double CentroidSideY;
    int nA, nB, idxA, idxB;
    double xA, xB;
    double yA, yB;

    double etaX, etaY;
    double zhiX, zhiY;

    double val;
    
    for (int i = 0; i < ElementNum; i++)
    {

        // ---------------- FILLING IN ELEMENT DATA ----------------
        E.ElementTag[i] = ElementTags[i];

        CentroidX = 0.0;
        CentroidY = 0.0;
        S = 0.0;

        for (int j = 0; j < 3; j++)
        {
            NodeTagBuffer[j] = NodeTags[3 * i + j];
            index = nodeIndex[NodeTagBuffer[j]];

            nodeToElements[NodeTagBuffer[j]].push_back(i);

            CentroidX += coords[3 * index];
            CentroidY += coords[3 * index + 1];

            if (j >= 1)
            {
                Xdist = NodeCoordBuffer[2 * j] - NodeCoordBuffer[2 * (j - 1)];
                Ydist = NodeCoordBuffer[2 * j + 1] - NodeCoordBuffer[2 * (j - 1) + 1];
                SideLengthBuffer[j - 1] = std::pow((Xdist) * (Xdist) + (Ydist) * (Ydist), 0.5);
                S += SideLengthBuffer[j - 1];
            }
            
            E.NodeTags[3 * i + j] = NodeTagBuffer[j];
            E.NodeCoords[6 * i + 2 * j] = coords[3 * index];
            E.NodeCoords[6 * i + 2 * j + 1] = coords[3 * index + 1];
        }

        CentroidX /= 3.0;
        CentroidY /= 3.0;

        E.CentroidCoords[2 * i] = CentroidX;
        E.CentroidCoords[2 * i + 1] = CentroidY;

        Xdist = NodeCoordBuffer[2 * 2] - NodeCoordBuffer[2 * 0];
        Ydist = NodeCoordBuffer[2 * 2 + 1] - NodeCoordBuffer[2 * 0 + 1];
        SideLengthBuffer[2] = std::pow((Xdist) * (Xdist) + (Ydist) * (Ydist), 0.5);
        S += SideLengthBuffer[2];
        S /= 2;

        // ---------------- DONE WITH FILLING IN ELEMENT DATA ----------------

        // ---------------- FILLING IN BOUNDARY DATA ----------------
        for(int j = 0; j < 3; j++)
        {
            // get node indices
            nA = E.NodeTags[3 * i + j];
            nB = E.NodeTags[3 * i + (j + 1) % 3];

            idxA = nodeIndex[nA];
            idxB = nodeIndex[nB];

            xA = coords[3 * idxA];
            yA = coords[3 * idxA + 1];

            xB = coords[3 * idxB];
            yB = coords[3 * idxB + 1];

            // LEFT: x = 0
            if(std::abs(xA) < tol && std::abs(xB) < tol)
            {
                E.IsLeft[3 * i + j] = 1;

                CentroidSideX = (xA + xB) / 2.0;
                CentroidSideY = (yA + yB) / 2.0;

                etaX = xB - xA;
                etaY = yB - yA;

                E.LeftFaceArea[6 * i + 2 * j] = etaY;
                E.LeftFaceArea[6 * i + 2 * j + 1] = -etaX;

                zhiX = CentroidSideX - CentroidX;
                zhiY = CentroidSideY - CentroidY;

                val = std::pow(etaX * etaX + etaY * etaY, 0.5);
                etaX = etaX / val;
                etaY = etaY / val;
                E.LeftDeltaEta[3 * i + j] = val;
                E.LeftEta_cap[6 * i + 2 * j] = etaX;
                E.LeftEta_cap[6 * i + 2 * j + 1] = etaY;

                val = std::pow(zhiX * zhiX + zhiY * zhiY, 0.5);
                zhiX = zhiX / val;
                zhiY = zhiY / val;
                E.LeftDeltaZhi[3 * i + j] = val;
                E.LeftZhi_cap[6 * i + 2 * j] = zhiX;
                E.LeftZhi_cap[6 * i + 2 * j + 1] = zhiY;

                leftNodes.insert(nA);
                leftNodes.insert(nB);
            }

            // RIGHT: x = 1
            if(std::abs(xA - 1.0) < tol && std::abs(xB - 1.0) < tol)
            {
                E.IsRight[3 * i + j] = 1;

                CentroidSideX = (xA + xB) / 2.0;
                CentroidSideY = (yA + yB) / 2.0;

                etaX = xB - xA;
                etaY = yB - yA;

                E.RightFaceArea[6 * i + 2 * j] = etaY;
                E.RightFaceArea[6 * i + 2 * j + 1] = -etaX;

                zhiX = CentroidSideX - CentroidX;
                zhiY = CentroidSideY - CentroidY;

                val = std::pow(etaX * etaX + etaY * etaY, 0.5);
                etaX = etaX / val;
                etaY = etaY / val;
                E.RightDeltaEta[3 * i + j] = val;
                E.RightEta_cap[6 * i + 2 * j] = etaX;
                E.RightEta_cap[6 * i + 2 * j + 1] = etaY;

                val = std::pow(zhiX * zhiX + zhiY * zhiY, 0.5);
                zhiX = zhiX / val;
                zhiY = zhiY / val;
                E.RightDeltaZhi[3 * i + j] = val;
                E.RightZhi_cap[6 * i + 2 * j] = zhiX;
                E.RightZhi_cap[6 * i + 2 * j + 1] = zhiY;

                rightNodes.insert(nA);
                rightNodes.insert(nB);
            }

            // BOTTOM: y = 0
            if(std::abs(yA) < tol && std::abs(yB) < tol)
            {
                E.IsBottom[3 * i + j] = 1;

                CentroidSideX = (xA + xB) / 2.0;
                CentroidSideY = (yA + yB) / 2.0;

                etaX = xB - xA;
                etaY = yB - yA;

                E.BottomFaceArea[6 * i + 2 * j] = etaY;
                E.BottomFaceArea[6 * i + 2 * j + 1] = -etaX;

                zhiX = CentroidSideX - CentroidX;
                zhiY = CentroidSideY - CentroidY;

                val = std::pow(etaX * etaX + etaY * etaY, 0.5);
                etaX = etaX / val;
                etaY = etaY / val;
                E.BottomDeltaEta[3 * i + j] = val;
                E.BottomEta_cap[6 * i + 2 * j] = etaX;
                E.BottomEta_cap[6 * i + 2 * j + 1] = etaY;

                val = std::pow(zhiX * zhiX + zhiY * zhiY, 0.5);
                zhiX = zhiX / val;
                zhiY = zhiY / val;
                E.BottomDeltaZhi[3 * i + j] = val;
                E.BottomZhi_cap[6 * i + 2 * j] = zhiX;
                E.BottomZhi_cap[6 * i + 2 * j + 1] = zhiY;

                bottomNodes.insert(nA);
                bottomNodes.insert(nB);
            }

            // TOP: y = 1
            if(std::abs(yA - 1.0) < tol && std::abs(yB - 1.0) < tol)
            {
                E.IsTop[3 * i + j] = 1;

                CentroidSideX = (xA + xB) / 2.0;
                CentroidSideY = (yA + yB) / 2.0;

                etaX = xB - xA;
                etaY = yB - yA;

                E.TopFaceArea[6 * i + 2 * j] = etaY;
                E.TopFaceArea[6 * i + 2 * j + 1] = -etaX;

                zhiX = CentroidSideX - CentroidX;
                zhiY = CentroidSideY - CentroidY;

                val = std::pow(etaX * etaX + etaY * etaY, 0.5);
                etaX = etaX / val;
                etaY = etaY / val;
                E.TopDeltaEta[3 * i + j] = val;
                E.TopEta_cap[6 * i + 2 * j] = etaX;
                E.TopEta_cap[6 * i + 2 * j + 1] = etaY;

                val = std::pow(zhiX * zhiX + zhiY * zhiY, 0.5);
                zhiX = zhiX / val;
                zhiY = zhiY / val;
                E.TopDeltaZhi[3 * i + j] = val;
                E.TopZhi_cap[6 * i + 2 * j] = zhiX;
                E.TopZhi_cap[6 * i + 2 * j + 1] = zhiY;

                topNodes.insert(nA);
                topNodes.insert(nB);
            }   
        }
        // ---------------- DONE WITH FILLING IN BOUNDARY DATA ----------------
    }
    
    // ---------------- NEIGHBOR DETECTION ----------------

    // buffer variables
    int n1, n2, n3;
    int a, b;
    int otherElem, otherEdge;

    std::map<std::pair<int,int>, std::pair<int,int>> edgeMap;
    
    for(int i = 0; i < ElementNum; i++)
    {
        n1 = E.NodeTags[3 * i];
        n2 = E.NodeTags[3 * i + 1];
        n3 = E.NodeTags[3 * i + 2];

        int edges[3][2] = {
            {n1, n2},
            {n2, n3},
            {n3, n1}
        };

        for(int e = 0; e < 3; e++)
        {
            a = edges[e][0];
            b = edges[e][1];

            if (a > b)
            {
                std::swap(a,b);
            }

            std::pair<int,int> key = {a,b};

            auto it = edgeMap.find(key);

            if( it == edgeMap.end())
            {
                edgeMap[key] = {i, e};
            }
            else
            {
                otherElem = it->second.first;
                otherEdge = it->second.second;

                E.NeighborElementTags[3 * i + e] = E.ElementTag[otherElem];
                E.NeighborElementTags[3 * otherElem + otherEdge] = E.ElementTag[i];
            }
        }
    }

    // ---------------- DONE WITH NEIGHBOR DETECTION ----------------

    // buffer variables
    double cx_i, cy_i;
    int neighbor;
    double DeltaEta, DeltaZhi;

    // ---------------- FILLING IN NEIGHBOR DATA ----------------
    for(int i = 0; i < ElementNum; i++)
    {
        cx_i = E.CentroidCoords[2 * i];
        cy_i = E.CentroidCoords[2 * i + 1];

        for(int j = 0; j < 3; j++)
        {
            neighbor = E.NeighborElementTags[3 * i + j];

            if(neighbor == 0)
            {
                continue; // skip boundary
            } 

            // ----------- EDGE NODES -----------
            nA = E.NodeTags[3 * i + j];
            nB = E.NodeTags[3 * i + (j + 1) % 3];

            idxA = nodeIndex[nA];
            idxB = nodeIndex[nB];

            xA = coords[3 * idxA];
            yA = coords[3 * idxA + 1];

            xB = coords[3 * idxB];
            yB = coords[3 * idxB + 1];

            // ----------- EDGE VECTOR (ETA) -----------
            etaX = xB - xA;
            etaY = yB - yA;

            DeltaEta = std::sqrt(etaX * etaX + etaY * etaY);

            etaX /= DeltaEta;
            etaY /= DeltaEta;

            E.NeighborEta_cap[6 * i + 2 * j]     = etaX;
            E.NeighborEta_cap[6 * i + 2 * j + 1] = etaY;
            E.NeighborDeltaEta[3 * i + j]      = DeltaEta;

            // ----------- FACE AREA VECTOR (NORMAL) -----------

            E.NeighborFaceArea[6 * i + 2 * j]     = etaY * DeltaEta;
            E.NeighborFaceArea[6 * i + 2 * j + 1] = -etaX * DeltaEta;

            // ----------- ZHI (centroid to centroid) -----------
            int idx_nb = elementIndex[neighbor];

            double cx_nb = E.CentroidCoords[2 * idx_nb];
            double cy_nb = E.CentroidCoords[2 * idx_nb + 1];

            double zhiX = cx_nb - cx_i;
            double zhiY = cy_nb - cy_i;

            DeltaZhi = std::sqrt(zhiX * zhiX + zhiY * zhiY);

            zhiX /= DeltaZhi;
            zhiY /= DeltaZhi;

            E.NeighborZhi_cap[6 * i + 2 * j]     = zhiX;
            E.NeighborZhi_cap[6 * i + 2 * j + 1] = zhiY;
            E.NeighborDeltaZhi[3 * i + j]      = DeltaZhi;
        }
    }

    // ---------------- DONE FILLING IN NEIGHBOR DATA ----------------

    std::vector<double> A(ElementNum * ElementNum, 0.0);
    std::vector<double> B(ElementNum, 0.0);
    std::vector<double> T(ElementNum, 100.0);
    std::vector<double> T_exact(ElementNum, 0.0);

    //buffer variables
    int idNeighbor;
    double PGcoeff, SGcoeff;
    std::vector<double> Af(2, 0.0);
    std::vector<double> EtaCap(2, 0.0);
    std::vector<double> ZhiCap(2, 0.0);
    std::vector<int> elems;

    //number for analytical
    int n = 100;

    for (int i = 0; i < ElementNum; i++)
    {
        T_exact[i] = T_analytical(E.CentroidCoords[2 * i], E.CentroidCoords[2 * i + 1], n);
        for (int j = 0; j < 3; j++)
        {
            neighbor = E.NeighborElementTags[3 * i + j];

            // ---------------- NEIGHBOR FACES ----------------
            if(neighbor != 0)
            {
                idNeighbor = elementIndex[neighbor];

                Af[0] = E.NeighborFaceArea[6 * i + 2 * j];
                Af[1] = E.NeighborFaceArea[6 * i + 2 * j + 1];

                EtaCap[0] = E.NeighborEta_cap[6 * i + 2 * j];
                EtaCap[1] = E.NeighborEta_cap[6 * i + 2 * j + 1];

                ZhiCap[0] = E.NeighborZhi_cap[6 * i + 2 * j];
                ZhiCap[1] = E.NeighborZhi_cap[6 * i + 2 * j + 1];

                DeltaEta = E.NeighborDeltaEta[3 * i + j];
                DeltaZhi = E.NeighborDeltaZhi[3 * i + j];

                PGcoeff = (Af[0] * Af[0] + Af[1] * Af[1]) / ((Af[0] * ZhiCap[0] + Af[1] * ZhiCap[1]) * DeltaZhi);
                SGcoeff = -(Af[0] * Af[0] + Af[1] * Af[1]) / ((Af[0] * ZhiCap[0] + Af[1] * ZhiCap[1]) * DeltaEta) * (EtaCap[0] * ZhiCap[0] + EtaCap[1] * ZhiCap[1]);

                A[i * ElementNum + i] -= PGcoeff;
                A[i * ElementNum + idNeighbor] += PGcoeff;

                nA = E.NodeTags[3 * i + j];
                nB = E.NodeTags[3 * i + (j + 1) % 3];

                if (leftNodes.count(nA))
                {
                    B[i] += SGcoeff * T_left;
                }
                
                else if (rightNodes.count(nA))
                {
                    B[i] += SGcoeff * T_right;
                }

                else if (topNodes.count(nA))
                {
                    B[i] += SGcoeff * T_top;
                }

                else if (bottomNodes.count(nA))
                {
                    B[i] += SGcoeff * T_bottom;
                }

                else
                {
                    elems = nodeToElements[nA];

                    for (int element = 0; element < elems.size(); element++)
                    {
                        A[i * ElementNum + elems[element]] -= (SGcoeff / elems.size());
                    }
                }

                if (leftNodes.count(nB))
                {
                    B[i] -= SGcoeff * T_left;
                }
                
                else if (rightNodes.count(nB))
                {
                    B[i] -= SGcoeff * T_right;
                }

                else if (topNodes.count(nB))
                {
                    B[i] -= SGcoeff * T_top;
                }

                else if (bottomNodes.count(nB))
                {
                    B[i] -= SGcoeff * T_bottom;
                }

                else
                {
                    elems = nodeToElements[nB];

                    for (int element = 0; element < elems.size(); element++)
                    {
                        A[i * ElementNum + elems[element]] += (SGcoeff / elems.size());
                    }
                }
            }
            // ---------------- DONE WITH NEIGHBOR FACES ----------------

            // ---------------- BOUNDARY FACES ----------------
            if (E.IsLeft[3 * i + j] == 1)
            {
                Af[0] = E.LeftFaceArea[6 * i + 2 * j];
                Af[1] = E.LeftFaceArea[6 * i + 2 * j + 1];

                EtaCap[0] = E.LeftEta_cap[6 * i + 2 * j];
                EtaCap[1] = E.LeftEta_cap[6 * i + 2 * j + 1];

                ZhiCap[0] = E.LeftZhi_cap[6 * i + 2 * j];
                ZhiCap[1] = E.LeftZhi_cap[6 * i + 2 * j + 1];

                DeltaEta = E.LeftDeltaEta[3 * i + j];
                DeltaZhi = E.LeftDeltaZhi[3 * i + j];

                PGcoeff = (Af[0] * Af[0] + Af[1] * Af[1]) / ((Af[0] * ZhiCap[0] + Af[1] * ZhiCap[1]) * DeltaZhi);

                B[i] -= PGcoeff * T_left;
                A[i * ElementNum + i] -= PGcoeff;
            }

            else if (E.IsRight[3 * i + j] == 1)
            {
                Af[0] = E.RightFaceArea[6 * i + 2 * j];
                Af[1] = E.RightFaceArea[6 * i + 2 * j + 1];

                EtaCap[0] = E.RightEta_cap[6 * i + 2 * j];
                EtaCap[1] = E.RightEta_cap[6 * i + 2 * j + 1];

                ZhiCap[0] = E.RightZhi_cap[6 * i + 2 * j];
                ZhiCap[1] = E.RightZhi_cap[6 * i + 2 * j + 1];

                DeltaEta = E.RightDeltaEta[3 * i + j];
                DeltaZhi = E.RightDeltaZhi[3 * i + j];

                PGcoeff = (Af[0] * Af[0] + Af[1] * Af[1]) / ((Af[0] * ZhiCap[0] + Af[1] * ZhiCap[1]) * DeltaZhi);

                B[i] -= PGcoeff * T_right;
                A[i * ElementNum + i] -= PGcoeff;
            }

            else if (E.IsTop[3 * i + j] == 1)
            {
                Af[0] = E.TopFaceArea[6 * i + 2 * j];
                Af[1] = E.TopFaceArea[6 * i + 2 * j + 1];

                EtaCap[0] = E.TopEta_cap[6 * i + 2 * j];
                EtaCap[1] = E.TopEta_cap[6 * i + 2 * j + 1];

                ZhiCap[0] = E.TopZhi_cap[6 * i + 2 * j];
                ZhiCap[1] = E.TopZhi_cap[6 * i + 2 * j + 1];

                DeltaEta = E.TopDeltaEta[3 * i + j];
                DeltaZhi = E.TopDeltaZhi[3 * i + j];

                PGcoeff = (Af[0] * Af[0] + Af[1] * Af[1]) / ((Af[0] * ZhiCap[0] + Af[1] * ZhiCap[1]) * DeltaZhi);

                B[i] -= PGcoeff * T_top;
                A[i * ElementNum + i] -= PGcoeff;
            }

            else if (E.IsBottom[3 * i + j] == 1)
            {
                Af[0] = E.BottomFaceArea[6 * i + 2 * j];
                Af[1] = E.BottomFaceArea[6 * i + 2 * j + 1];

                EtaCap[0] = E.BottomEta_cap[6 * i + 2 * j];
                EtaCap[1] = E.BottomEta_cap[6 * i + 2 * j + 1];

                ZhiCap[0] = E.BottomZhi_cap[6 * i + 2 * j];
                ZhiCap[1] = E.BottomZhi_cap[6 * i + 2 * j + 1];

                DeltaEta = E.BottomDeltaEta[3 * i + j];
                DeltaZhi = E.BottomDeltaZhi[3 * i + j];

                PGcoeff = (Af[0] * Af[0] + Af[1] * Af[1]) / ((Af[0] * ZhiCap[0] + Af[1] * ZhiCap[1]) * DeltaZhi);

                B[i] -= PGcoeff * T_bottom;
                A[i * ElementNum + i] -= PGcoeff;
            }
            // ---------------- DONE WITH BOUNDARY FACES ----------------
        }
    }
    
    // ---------------- GAUSS-SEIDEL SOLVER ----------------

    int maxIter = 10000;
    double tolGS = 1e-10;

    for(int iter = 0; iter < maxIter; iter++)
    {
        double maxError = 0.0;

        for(int i = 0; i < ElementNum; i++)
        {
            double sigma = 0.0;

            for(int j = 0; j < ElementNum; j++)
            {
                if(j != i)
                {
                    sigma += A[i * ElementNum + j] * T[j];
                }
            }

            double T_old = T[i];
            T[i] = (B[i] - sigma) / A[i * ElementNum + i];

            maxError = std::max(maxError, std::abs(T[i] - T_old));
        }

        if(maxError < tolGS)
        {
            std::cout << "Converged in " << iter << " iterations\n";
            break;
        }

        if(iter == maxIter - 1)
        {
            std::cout << "Did not converge\n";
        }
    }

    // ---------------- ERROR ANALYSIS ----------------

    double maxDiff = 0.0;

    for(int i = 0; i < ElementNum; i++)
    {
        double diff = std::abs(T[i] - T_exact[i]);

        std::cout << "Elem " << i
                << " | Numerical: " << T[i]
                << " | Exact: " << T_exact[i]
                << " | Diff: " << diff << "\n";

        if(diff > maxDiff)
            maxDiff = diff;
    }

    std::cout << "\nMax difference: " << maxDiff << "\n";

    return 0;


}