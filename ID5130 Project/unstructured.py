import gmsh
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
gmsh.initialize()

gmsh.model.add("square")

lc = 0.15  # Characteristic length

# Define points
gmsh.model.geo.addPoint(0, 0, 0, lc, 1)
gmsh.model.geo.addPoint(1, 0, 0, lc, 2)
gmsh.model.geo.addPoint(1, 1, 0, lc, 3)
gmsh.model.geo.addPoint(0, 1, 0, lc, 4)

# Define lines
gmsh.model.geo.addLine(1, 2, 1)
gmsh.model.geo.addLine(2, 3, 2)
gmsh.model.geo.addLine(3, 4, 3)
gmsh.model.geo.addLine(4, 1, 4)

# Create curve loop and surface
gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)
gmsh.model.geo.addPlaneSurface([1], 1)

gmsh.model.geo.synchronize()

# Generate 2D mesh
gmsh.model.mesh.generate(2)

# Write mesh to file
gmsh.write("square.msh")

types, elementTags, nodeTags = gmsh.model.mesh.getElements()
node_tags_for_vertices, coords, _ = gmsh.model.mesh.getNodes()

triangle_index = list(types).index(2)

gmsh.finalize()

R = np.array([[0, 1], [-1, 0]])

elementTags = list(elementTags[triangle_index])
nodeTags = list(nodeTags[triangle_index])

node_coords = []
centroids = []
neighbors = {}

left_boundary = {}
right_boundary = {}
top_boundary = {}
bottom_boundary = {}
volume = []

nodes_common = {}

node_element = []


def get_node_coords(node_id, node_tags, coords):
    
    index = list(node_tags).index(node_id)
    x = coords[3 * index]
    y = coords[3 * index + 1]
    return np.array([x, y])


for i in node_tags_for_vertices:
    
    e_list = []
    indices = [k for k, val in enumerate(nodeTags) if val == i]
    
    for j in indices:
        
        ind = int((j - j%3) / 3)
        e_list.append(elementTags[ind])
    nodes_common[i] = e_list


for i in range(len(elementTags)):
    
    n1, n2, n3 = nodeTags[3 * i], nodeTags[3 * i + 1], nodeTags[3 * i + 2]
    
    coord_array = np.array((get_node_coords(n1, node_tags_for_vertices, coords), get_node_coords(n2, node_tags_for_vertices, coords), get_node_coords(n3, node_tags_for_vertices, coords)))
    node_coords.append(coord_array)
    
    centroid_x = (coord_array[0][0] + coord_array[1][0] + coord_array[2][0]) / 3
    centroid_y = (coord_array[0][1] + coord_array[1][1] + coord_array[2][1]) / 3
    centroid = np.array((centroid_x, centroid_y))
    centroids.append(centroid)

    x_list = [coord_array[0][0], coord_array[1][0], coord_array[2][0]]
    y_list = [coord_array[0][1], coord_array[1][1], coord_array[2][1]]

    a = ((x_list[0] - x_list[1]) ** 2 + (y_list[0] - y_list[1]) ** 2) ** 0.5
    b = ((x_list[2] - x_list[1]) ** 2 + (y_list[2] - y_list[1]) ** 2) ** 0.5
    c = ((x_list[0] - x_list[2]) ** 2 + (y_list[0] - y_list[2]) ** 2) ** 0.5

    s = (a + b + c) / 2

    volume.append((s * (s - a) * (s - b) * (s - c)) ** 0.5)
    
    n = [n1, n2, n3]

    neighbor_per_element = []
    
    for j in elementTags:

        element_return = {}
        
        if j != elementTags[i]:
            
            index = elementTags.index(j)
            node_tuple = (nodeTags[3 * index], nodeTags[3 * index + 1], nodeTags[3 * index + 2])

            coord_array_neighbor = [get_node_coords(node_tuple[0], node_tags_for_vertices, coords), get_node_coords(node_tuple[1], node_tags_for_vertices, coords), get_node_coords(node_tuple[2], node_tags_for_vertices, coords)]
            centroid_neighbor = sum(coord_array_neighbor) / 3
            
            if n1 in node_tuple and n2 in node_tuple:
                
                element_return["element"] = j
                element_return["eta_cap"] = (coord_array[1] - coord_array[0]) / np.linalg.norm((coord_array[1] - coord_array[0]))
                element_return["area"] = np.dot(R, (coord_array[1] - coord_array[0]))
                element_return["delta_eta"] = np.linalg.norm((coord_array[1] - coord_array[0]))
                element_return["zhi_cap"] = (centroid_neighbor - centroid) / np.linalg.norm((centroid_neighbor - centroid))
                element_return["delta_zhi"] = np.linalg.norm((centroid_neighbor - centroid))
                element_return["n1"] = [n1, nodes_common[n1]]
                element_return["n2"] = [n2, nodes_common[n2]]

                neighbor_per_element.append(element_return)
                
            if n2 in node_tuple and n3 in node_tuple:
                
                element_return["element"] = j
                element_return["eta_cap"] = (coord_array[2] - coord_array[1]) / np.linalg.norm((coord_array[2] - coord_array[1]))
                element_return["area"] = np.dot(R, (coord_array[2] - coord_array[1]))
                element_return["delta_eta"] = np.linalg.norm((coord_array[2] - coord_array[1]))
                element_return["zhi_cap"] = (centroid_neighbor - centroid) / np.linalg.norm((centroid_neighbor - centroid))
                element_return["delta_zhi"] = np.linalg.norm((centroid_neighbor - centroid))
                element_return["n1"] = [n2, nodes_common[n2]]
                element_return["n2"] = [n3, nodes_common[n3]]

                neighbor_per_element.append(element_return)
                
            if n3 in node_tuple and n1 in node_tuple:
                
                element_return["element"] = j
                element_return["eta_cap"] = (coord_array[0] - coord_array[2]) / np.linalg.norm((coord_array[0] - coord_array[2]))
                element_return["area"] = np.dot(R, (coord_array[0] - coord_array[2]))
                element_return["delta_eta"] = np.linalg.norm((coord_array[0] - coord_array[2]))
                element_return["zhi_cap"] = (centroid_neighbor - centroid) / np.linalg.norm((centroid_neighbor - centroid))
                element_return["delta_zhi"] = np.linalg.norm((centroid_neighbor - centroid))
                element_return["n1"] = [n3, nodes_common[n3]]
                element_return["n2"] = [n1, nodes_common[n1]]

                neighbor_per_element.append(element_return)

            
                
    neighbors[elementTags[i]] = neighbor_per_element
    
    left = {}
    right = {}
    top = {}
    bottom = {}
    
    if x_list.count(0) == 2:
        
        indices = [i for i, val in enumerate(x_list) if val == 0]
        
        if len(indices) != 0:
            
            if max(indices) - min(indices) > 1:
                
                node_1_coords = get_node_coords(n[indices[1]], node_tags_for_vertices, coords)
                node_2_coords = get_node_coords(n[indices[0]], node_tags_for_vertices, coords)
                l1 = [n[indices[1]], nodes_common[n[indices[1]]]]
                l2 = [n[indices[0]], nodes_common[n[indices[0]]]]
                
            else:
                
                node_1_coords = get_node_coords(n[indices[0]], node_tags_for_vertices, coords)
                node_2_coords = get_node_coords(n[indices[1]], node_tags_for_vertices, coords)
                l1 = [n[indices[0]], nodes_common[n[indices[0]]]]
                l2 = [n[indices[1]], nodes_common[n[indices[1]]]]

            centroid_face = (node_1_coords + node_2_coords) / 2

            left["zhi_cap"] = (centroid_face - centroid) / np.linalg.norm(centroid_face - centroid)
            left["delta_zhi"] = np.linalg.norm(centroid_face - centroid)
            left["eta_cap"] = (node_2_coords - node_1_coords) / np.linalg.norm(node_2_coords - node_1_coords)
            left["delta_eta"] = np.linalg.norm(node_2_coords - node_1_coords)
            left["area"] = np.dot(R, node_2_coords - node_1_coords)
            left["n1"] = l1
            left["n2"] = l2
             
    if x_list.count(1) == 2:
        
        indices = [i for i, val in enumerate(x_list) if val == 1]
        
        if len(indices) != 0:
            
            if max(indices) - min(indices) > 1:
                
                node_1_coords = get_node_coords(n[indices[1]], node_tags_for_vertices, coords)
                node_2_coords = get_node_coords(n[indices[0]], node_tags_for_vertices, coords)
                l1 = [n[indices[1]], nodes_common[n[indices[1]]]]
                l2 = [n[indices[0]], nodes_common[n[indices[0]]]]
                
            else:
                
                node_1_coords = get_node_coords(n[indices[0]], node_tags_for_vertices, coords)
                node_2_coords = get_node_coords(n[indices[1]], node_tags_for_vertices, coords)
                l1 = [n[indices[0]], nodes_common[n[indices[0]]]]
                l2 = [n[indices[1]], nodes_common[n[indices[1]]]]

            centroid_face = (node_1_coords + node_2_coords) / 2

            right["zhi_cap"] = (centroid_face - centroid) / np.linalg.norm(centroid_face - centroid)
            right["delta_zhi"] = np.linalg.norm(centroid_face - centroid)
            right["eta_cap"] = (node_2_coords - node_1_coords) / np.linalg.norm(node_2_coords - node_1_coords)
            right["delta_eta"] = np.linalg.norm(node_2_coords - node_1_coords)
            right["area"] = np.dot(R, node_2_coords - node_1_coords)
            right["n1"] = l1
            right["n2"] = l2
                
    if y_list.count(0) == 2:
        
        indices = [i for i, val in enumerate(y_list) if val == 0]
        
        if len(indices) != 0:
            
            if max(indices) - min(indices) > 1:
                
                node_1_coords = get_node_coords(n[indices[1]], node_tags_for_vertices, coords)
                node_2_coords = get_node_coords(n[indices[0]], node_tags_for_vertices, coords)
                l1 = [n[indices[1]], nodes_common[n[indices[1]]]]
                l2 = [n[indices[0]], nodes_common[n[indices[0]]]]
                
            else:
                
                node_1_coords = get_node_coords(n[indices[0]], node_tags_for_vertices, coords)
                node_2_coords = get_node_coords(n[indices[1]], node_tags_for_vertices, coords)
                l1 = [n[indices[0]], nodes_common[n[indices[0]]]]
                l2 = [n[indices[1]], nodes_common[n[indices[1]]]]

            centroid_face = (node_1_coords + node_2_coords) / 2

            bottom["zhi_cap"] = (centroid_face - centroid) / np.linalg.norm(centroid_face - centroid)
            bottom["delta_zhi"] = np.linalg.norm(centroid_face - centroid)
            bottom["eta_cap"] = (node_2_coords - node_1_coords) / np.linalg.norm(node_2_coords - node_1_coords)
            bottom["delta_eta"] = np.linalg.norm(node_2_coords - node_1_coords)
            bottom["area"] = np.dot(R, node_2_coords - node_1_coords)
            bottom["n1"] = l1
            bottom["n2"] = l2
                
    if y_list.count(1) == 2:
        
        indices = [i for i, val in enumerate(y_list) if val == 1]
        
        if len(indices) != 0:
            
            if max(indices) - min(indices) > 1:
                
                node_1_coords = get_node_coords(n[indices[1]], node_tags_for_vertices, coords)
                node_2_coords = get_node_coords(n[indices[0]], node_tags_for_vertices, coords)
                l1 = [n[indices[1]], nodes_common[n[indices[1]]]]
                l2 = [n[indices[0]], nodes_common[n[indices[0]]]]
                
            else:
                
                node_1_coords = get_node_coords(n[indices[0]], node_tags_for_vertices, coords)
                node_2_coords = get_node_coords(n[indices[1]], node_tags_for_vertices, coords)
                l1 = [n[indices[0]], nodes_common[n[indices[0]]]]
                l2 = [n[indices[1]], nodes_common[n[indices[1]]]]

            centroid_face = (node_1_coords + node_2_coords) / 2

            top["zhi_cap"] = (centroid_face - centroid) / np.linalg.norm(centroid_face - centroid)
            top["delta_zhi"] = np.linalg.norm(centroid_face - centroid)
            top["eta_cap"] = (node_2_coords - node_1_coords) / np.linalg.norm(node_2_coords - node_1_coords)
            top["delta_eta"] = np.linalg.norm(node_2_coords - node_1_coords)
            top["area"] = np.dot(R, node_2_coords - node_1_coords)
            top["n1"] = l1
            top["n2"] = l2

    left_boundary[elementTags[i]] = left
    right_boundary[elementTags[i]] = right
    top_boundary[elementTags[i]] = top
    bottom_boundary[elementTags[i]] = bottom

def G(element, neighbors, left_boundary, right_boundary, top_boundary, bottom_boundary, phi_prev, elementTags, k):

    dict_neighbor = {}

    dict_left = {}
    dict_right = {}
    dict_top = {}
    dict_bottom = {}
    
    for i in neighbors[element]:

        phi1 = 0
        phi2 = 0
        l1 = i["n1"][1]
        l2 = i["n2"][1]

        for j in l1:
            
            phi_index = elementTags.index(j)
            phi1 += phi_prev[phi_index]

        for j in l2:
            
            phi_index = elementTags.index(j)
            phi2 += phi_prev[phi_index]

        phi1 /= len(l1)
        phi2 /= len(l2)

        el1 = ((np.dot(i["area"], i["area"])) / (np.dot(i["area"], i["zhi_cap"]))) / i["delta_zhi"] * k
        el2 = ((phi2 - phi1) / i["delta_eta"]) * ((np.dot(i["area"], i["area"])) / (np.dot(i["area"], i["zhi_cap"]))) * np.dot(i["eta_cap"], i["zhi_cap"]) * k

        dict_neighbor[i["element"]] = [el1, 0]

    left = left_boundary[element]
    
    if len(left) != 0:

        phi1 = 0
        phi2 = 0
        l1 = left["n1"][1]
        l2 = left["n2"][1]

        for j in l1:
            
            phi_index = elementTags.index(j)
            phi1 += phi_prev[phi_index]

        for j in l2:
            
            phi_index = elementTags.index(j)
            phi2 += phi_prev[phi_index]

        phi1 /= len(l1)
        phi2 /= len(l2)

        el1 = ((np.dot(left["area"], left["area"])) / (np.dot(left["area"], left["zhi_cap"]))) / left["delta_zhi"] * k
        #el2 = ((phi2 - phi1) / left["delta_eta"]) * ((np.dot(left["area"], left["area"])) / (np.dot(left["area"], left["zhi_cap"]))) * np.dot(left["eta_cap"], left["zhi_cap"]) * k

        dict_left[element] = [el1, 0]

    right = right_boundary[element]
        
    if len(right) != 0:

        phi1 = 0
        phi2 = 0
        l1 = right["n1"][1]
        l2 = right["n2"][1]

        for j in l1:
            
            phi_index = elementTags.index(j)
            phi1 += phi_prev[phi_index]

        for j in l2:
            
            phi_index = elementTags.index(j)
            phi2 += phi_prev[phi_index]

        phi1 /= len(l1)
        phi2 /= len(l2)

        el1 = ((np.dot(right["area"], right["area"])) / (np.dot(right["area"], right["zhi_cap"]))) / right["delta_zhi"] * k
        #el2 = ((phi2 - phi1) / right["delta_eta"]) * ((np.dot(right["area"], right["area"])) / (np.dot(right["area"], right["zhi_cap"]))) * np.dot(right["eta_cap"], right["zhi_cap"]) * k

        dict_right[element] = [el1, 0]

    top = top_boundary[element]
        
    if len(top) != 0:

        phi1 = 0
        phi2 = 0
        l1 = top["n1"][1]
        l2 = top["n2"][1]

        for j in l1:
            
            phi_index = elementTags.index(j)
            phi1 += phi_prev[phi_index]

        for j in l2:
            
            phi_index = elementTags.index(j)
            phi2 += phi_prev[phi_index]

        phi1 /= len(l1)
        phi2 /= len(l2)

        el1 = ((np.dot(top["area"], top["area"])) / (np.dot(top["area"], top["zhi_cap"]))) / top["delta_zhi"] * k
        #el2 = ((phi2 - phi1) / top["delta_eta"]) * ((np.dot(top["area"], top["area"])) / (np.dot(top["area"], top["zhi_cap"]))) * np.dot(top["eta_cap"], top["zhi_cap"]) * k

        dict_top[element] = [el1, 0]

    bottom = bottom_boundary[element]

    if len(bottom) != 0:

        phi1 = 0
        phi2 = 0
        l1 = bottom["n1"][1]
        l2 = bottom["n2"][1]

        for j in l1:
            
            phi_index = elementTags.index(j)
            phi1 += phi_prev[phi_index]

        for j in l2:
            
            phi_index = elementTags.index(j)
            phi2 += phi_prev[phi_index]

        phi1 /= len(l1)
        phi2 /= len(l2)

        el1 = ((np.dot(bottom["area"], bottom["area"])) / (np.dot(bottom["area"], bottom["zhi_cap"]))) / bottom["delta_zhi"] * k
        #el2 = ((phi2 - phi1) / bottom["delta_eta"]) * ((np.dot(bottom["area"], bottom["area"])) / (np.dot(bottom["area"], bottom["zhi_cap"]))) * np.dot(bottom["eta_cap"], bottom["zhi_cap"]) * k

        dict_bottom[element] = [el1, 0]


        

    return dict_neighbor, dict_left, dict_right, dict_top, dict_bottom

"""SOLVING THE EQUATIONS"""

k = 10

T_left = 50
T_right = 200
T_top = 100
T_bottom = 300

def Temp_analytical(x, y, n, T_left, T_right, T_top, T_bottom): # x and y are the coordinate vectors at which tempeature needs to be evaluated
    T = 0
    for i in range (1, n + 1):
        if i % 2 != 0:
            T += T_left * (4 / (i * np.pi)) * np.sin(i * np.pi * y) * (np.sinh(i * np.pi * (1 - x)) / np.sinh(i * np.pi))
            T += T_bottom * (4 / (i * np.pi)) * np.sin(i * np.pi * x) * (np.sinh(i * np.pi * (1 - y)) / np.sinh(i * np.pi))
            T += T_right * (4 / (i * np.pi)) * np.sin(i * np.pi * y) * (np.sinh(i * np.pi * x) / np.sinh(i * np.pi))
            T += T_top * (4 / (i * np.pi)) * np.sin(i * np.pi * x) * (np.sinh(i * np.pi * y) / np.sinh(i * np.pi))
    return T #returns a single value of T for one x and one y coordinate

tot = len(elementTags)

A = np.zeros([tot, tot])
B = np.zeros(tot)

phi = np.ones(tot) * 100
phi_prev = np.ones(tot) * 100

#print(G(39, neighbors, left_boundary, right_boundary, top_boundary, bottom_boundary, phi_prev, elementTags, 10))
#print(neighbors[39])

err = 0
count = 0

no_of_iterations = 300

flag = False

while count <= no_of_iterations:

    for i in elementTags:

        index_0_GS = elementTags.index(i)

        for j in elementTags:

            index_0 = elementTags.index(j)

            B[index_0] = 0

            dict_neighbor, dict_left, dict_right, dict_top, dict_bottom = G(j, neighbors, left_boundary, right_boundary, top_boundary, bottom_boundary, phi_prev, elementTags, k)

            PG_neighbor = {}
            SG_neighbor = {}

            for e in dict_neighbor:
                PG_neighbor[e] = dict_neighbor[e][0]
                SG_neighbor[e] = dict_neighbor[e][1]

            PG_left = 0
            PG_right = 0
            PG_top = 0
            PG_bottom = 0

            SG_left = 0
            SG_right = 0
            SG_top = 0
            SG_bottom = 0

            if len(dict_left) != 0:

                PG_left = dict_left[j][0]
                SG_left = dict_left[j][1]

            if len(dict_right) != 0:
                
                PG_right = dict_right[j][0]
                SG_right = dict_right[j][1]
                
            if len(dict_top) != 0:
                
                PG_top = dict_top[j][0]
                SG_top = dict_top[j][1]

            if len(dict_bottom) != 0:
                
                PG_bottom = dict_bottom[j][0]
                SG_bottom = dict_bottom[j][1]

            if flag == False:

                for l in PG_neighbor:

                     index_1 = elementTags.index(l)

                     A[index_0][index_0] -= PG_neighbor[l]

                     A[index_0][index_1] += PG_neighbor[l]

                A[index_0][index_0] += -(PG_left + PG_right + PG_top + PG_bottom)

            for l in SG_neighbor:
                
                B[index_0] += SG_neighbor[l]        
                
            B[index_0] += SG_left + SG_right + SG_top + SG_bottom - PG_left * T_left - PG_right * T_right - PG_top * T_top - PG_bottom * T_bottom
            
        flag = True

        phi[index_0_GS] = (B[index_0_GS] - (np.dot(A[index_0_GS], phi) - A[index_0_GS][index_0_GS] * phi[index_0_GS])) / A[index_0_GS][index_0_GS]

    
    err = sum(abs(phi - phi_prev)) / tot
    
    phi_prev = phi.copy()

    count += 1

    #print("Iteration:", count, "Error:", err)

    if err <= 0.01:
        break

x = []
y = []

for i in centroids:
    x.append(float(i[0]))
    y.append(float(i[1]))

T = list(phi)

# left bottom corner, left, left top corner, right bottom corner, right, right top corner, top, bottom

n = int(np.ceil(1 / lc))

x_add = []

x_add.extend(list(np.zeros(n + 1)))
x_add.extend(list(np.ones(n + 1)))
x_add.extend(list(np.linspace(0.15, 1 - 0.15, n - 1)))
x_add.extend(list(np.linspace(0.15, 1 - 0.15, n - 1)))

y_add = []

y_add.extend(list(np.linspace(0, 1, n + 1)))
y_add.extend(list(np.linspace(0, 1, n + 1)))
y_add.extend(list(np.ones(n - 1)))
y_add.extend(list(np.zeros(n - 1)))


T_add = [(T_left + T_bottom) / 2]

for i in range(n - 1):

    T_add.append(T_left)

T_add.append((T_left + T_top) / 2)

T_add.append((T_right + T_bottom) / 2)

for i in range(n - 1):

    T_add.append(T_right)

T_add.append((T_right + T_top) / 2)

for i in range(n - 1):

    T_add.append(T_top)

for i in range(n - 1):

    T_add.append(T_bottom)
    

x.extend(x_add)
y.extend(y_add)
T.extend(T_add)


x = np.array(x)
y = np.array(y)

T = np.array(T)

T_a = []

for i in range(len(x)):
    T_a.append(Temp_analytical(x[i], y[i], 100, T_left, T_right, T_top, T_bottom))

T_a = np.array(T_a)

error = (T - T_a)

xi = np.linspace(x.min(), x.max(), 100)
yi = np.linspace(y.min(), y.max(), 100)
xi, yi = np.meshgrid(xi, yi)

zi = griddata((x, y), T, (xi, yi), method='linear')

# Plot 2D heatmap
plt.figure(figsize=(6, 6))
plt.axis('equal')
plt.gca().set_aspect('equal', adjustable='box')
c = plt.pcolormesh(xi, yi, zi, shading='auto', cmap='inferno')
cbar = plt.colorbar(c)
cbar.set_label('Temperature (Unstructured)', fontsize=25)
plt.xlabel('X')
plt.ylabel('Y')
#plt.title("Temperature (Unstructured)")
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
cbar.ax.tick_params(labelsize=25)
plt.tight_layout()
plt.show()

zi = griddata((x, y), T_a, (xi, yi), method='linear')

# Plot 2D heatmap
plt.figure(figsize=(6, 6))
plt.axis('equal')
plt.gca().set_aspect('equal', adjustable='box')
c = plt.pcolormesh(xi, yi, zi, shading='auto', cmap='inferno')
cbar = plt.colorbar(c)
cbar.set_label('Temperature (Analytical)', fontsize=25)
plt.xlabel('X')
plt.ylabel('Y')
#plt.title("Temperature (Analytical)")
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
cbar.ax.tick_params(labelsize=25)
plt.tight_layout()
plt.show()

zi = griddata((x, y), error, (xi, yi), method='linear')

# Plot 2D heatmap
plt.figure(figsize=(6, 6))
plt.axis('equal')
plt.gca().set_aspect('equal', adjustable='box')
c = plt.pcolormesh(xi, yi, zi, shading='auto', cmap='viridis')
cbar = plt.colorbar(c)
cbar.set_label('Temperature (Error)', fontsize=25)
plt.xlabel('X')
plt.ylabel('Y')
#plt.title("Temperature (Error)")
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
cbar.ax.tick_params(labelsize=25)
plt.tight_layout()
plt.show()
