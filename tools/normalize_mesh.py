import bpy
import os.path as osp
import numpy as np

this_dir = osp.dirname(__file__)
root_path = osp.join(this_dir, '..', 'data', 'YCB_Video')

filename = osp.join(root_path, 'models', 'sanning_mug', 'mug.obj')
print(filename)

imported_object = bpy.ops.import_scene.obj(filepath=filename)
obj_object = bpy.context.selected_objects[0]
print('Imported object: ', obj_object.name)

# collect the vertices
vertices = np.zeros((0, 3), dtype=np.float32)
for item in bpy.data.objects:
    if item.type == 'MESH':
        for vertex in item.data.vertices:
            vertices = np.append(vertices, np.array(vertex.co).reshape((1,3)), axis = 0)

print(vertices.shape)

# switch y axis and z axis
i = 0
m = np.mean(vertices[:, 2])
for item in bpy.data.objects:
    if item.type == 'MESH':
        for vertex in item.data.vertices:
            rv = vertex.co
            vertex.co[0] = vertices[i, 0] * 0.01
            vertex.co[1] = vertices[i, 1] * 0.01
            vertex.co[2] = (vertices[i, 2] - m) * 0.01
            i += 1
       
# save model
bpy.data.objects[obj_object.name].select = True
filename = osp.join(root_path, 'models', 'sanning_mug', 'mug_normalized.obj')
bpy.ops.export_scene.obj(filepath=filename, use_selection=True)
