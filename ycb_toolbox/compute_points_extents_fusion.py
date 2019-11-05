import bpy
import os.path as osp
import numpy as np
#from ycb_globals import ycb_video

this_dir = osp.dirname(__file__)
root_path = osp.join(this_dir, '..', 'data', 'YCB_Object')
classes = ('fusion_duplo_dude', )

# extent file
filename = osp.join(root_path, 'models', 'extents_fusion.txt')
fext = open(filename, "w")

for i in range(len(classes)):

    # clear model
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_pattern(pattern="Camera")
    bpy.ops.object.select_all(action='INVERT')
    bpy.ops.object.delete()

    cls = classes[i]
    print(cls)
    filename = osp.join(root_path, 'models', cls, 'textured_simple.obj')
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

    # write extent
    extent = 2 * np.max(np.absolute(vertices), axis=0)
    print(extent)
    fext.write('%f %f %f\n' % (extent[0], extent[1], extent[2]))

    # write points
    num = 3000
    if vertices.shape[0] > num:
        perm = np.random.permutation(np.arange(vertices.shape[0]))
        index = perm[:3000]
        pcloud = vertices[index, :]
    else:
        factor = np.ceil(num / vertices.shape[0])
        pcloud = np.repeat(vertices, factor, axis=0)

    filename = osp.join(root_path, 'models', cls, 'points.xyz')
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
