import bpy
import os.path as osp
import numpy as np
from ycb_globals import ycb_video

this_dir = osp.dirname(__file__)
root_path = osp.join(this_dir, '..', 'data', 'YCB_Video')
opt = ycb_video()

classes = opt.classes + ('sanning_mug', )

# extent file
filename = osp.join(root_path, 'models_sim', 'extents.txt')
fext = open(filename, "w")

for i in range(len(classes)):

    # clear model
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_pattern(pattern="Camera")
    bpy.ops.object.select_all(action='INVERT')
    bpy.ops.object.delete()

    cls = classes[i]
    print(cls)
    filename = osp.join(root_path, 'models_sim', cls, 'meshes', cls + '.obj')
    print(filename)

    imported_object = bpy.ops.import_scene.obj(filepath=filename)
    obj_object = bpy.context.selected_objects[0]
    print('Imported object: ', obj_object.name)

    # collect the vertices
    vertices = np.zeros((0, 3), dtype=np.float32)
    for item in bpy.data.objects:
        if item.type == 'MESH':
            for vertex in item.data.vertices:
                vertices = np.append(vertices, np.array(vertex.co).reshape((1,3)), axis=0)

    print(vertices.shape)

    # normalization
    i = 0
    if cls == 'sanning_mug':
        m = np.mean(vertices[:, 1])
    else:
        m = 0

    if cls == '037_scissors':
        factor = 0.0001
    else:
        factor = 0.01

    vertices[:, 0] *= factor
    vertices[:, 1] -= m
    vertices[:, 1] *= factor
    vertices[:, 2] *= factor

    # write extent
    extent = 2 * np.max(np.absolute(vertices), axis=0)
    print(extent)
    fext.write('%f %f %f\n' % (extent[0], extent[1], extent[2]))

    # write points
    perm = np.random.permutation(np.arange(vertices.shape[0]))
    index = perm[:3000]
    pcloud = vertices[index, :]

    filename = osp.join(root_path, 'models_sim', cls, 'meshes', 'points.xyz')
    f = open(filename, "w")
    for i in range(pcloud.shape[0]):
        f.write('%f %f %f\n' % (pcloud[i, 0], pcloud[i, 1], pcloud[i, 2]))
    f.close()

    # clear model
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_pattern(pattern="Camera")
    bpy.ops.object.select_all(action='INVERT')
    bpy.ops.object.delete()

    # The meshes still present after delete
    for item in bpy.data.meshes:
        bpy.data.meshes.remove(item)
    for item in bpy.data.materials:
        bpy.data.materials.remove(item)

fext.close()
