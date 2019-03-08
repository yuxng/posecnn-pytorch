import bpy
import os.path as osp
import numpy as np

this_dir = osp.dirname(__file__)
root_path = osp.join(this_dir, '..', 'data', 'YCB_Video')
dirname = '001_chips_can'
modelname = '001_chips_can.obj'

filename = osp.join(root_path, 'models', dirname, modelname)
print(filename)

imported_object = bpy.ops.import_scene.obj(filepath=filename)
obj_object = bpy.context.selected_objects[0]
print('Imported object: ', obj_object.name)

# deselect all
bpy.ops.object.select_all(action='DESELECT')
bpy.data.objects['Cube'].select = True
bpy.ops.object.delete() 

# collect the vertices
vertices = np.zeros((0, 3), dtype=np.float32)
for item in bpy.data.objects:
    if item.type == 'MESH':
        for vertex in item.data.vertices:
            vertices = np.append(vertices, np.array(vertex.co).reshape((1,3)), axis = 0)

print(vertices.shape)

# process vertices
i = 0
mx = np.mean(vertices[:, 0])
my = np.mean(vertices[:, 1])
mz = np.mean(vertices[:, 2])
for item in bpy.data.objects:
    if item.type == 'MESH':
        for vertex in item.data.vertices:
            rv = vertex.co
            vertex.co[0] = vertices[i, 0] - mx
            vertex.co[1] = vertices[i, 1] - my
            vertex.co[2] = vertices[i, 2] - mz
            i += 1
       
# save model
bpy.ops.object.select_all(action='DESELECT')
#bpy.ops.object.select_pattern(pattern="Camera")
#bpy.ops.object.select_all(action='INVERT')
bpy.data.objects[obj_object.name].select = True
filename = osp.join(root_path, 'models', dirname, 'textured_simple.obj')
bpy.ops.export_scene.obj(filepath=filename, use_selection=True)

# write extent
vertices = np.zeros((0, 3), dtype=np.float32)
for item in bpy.data.objects:
    if item.type == 'MESH':
        for vertex in item.data.vertices:
            vertices = np.append(vertices, np.array(vertex.co).reshape((1,3)), axis = 0)

extent = 2 * np.max(np.absolute(vertices), axis=0)
print(extent)
filename = osp.join(root_path, 'models', dirname, 'extent.txt')
f = open(filename, "w")
f.write('%f %f %f' % (extent[0], extent[1], extent[2]))
f.close()

# write points
perm = np.random.permutation(np.arange(vertices.shape[0]))
index = perm[:3000]
pcloud = vertices[index, :]

filename = osp.join(root_path, 'models', dirname, 'points.xyz')
f = open(filename, "w")
for i in range(pcloud.shape[0]):
    f.write('%f %f %f\n' % (pcloud[i, 0], pcloud[i, 1], pcloud[i, 2]))
f.close()
