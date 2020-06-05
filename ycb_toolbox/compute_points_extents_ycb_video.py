import bpy
import os.path as osp
import numpy as np
#from ycb_globals import ycb_video

this_dir = osp.dirname(__file__)
root_path = osp.join(this_dir, '..', 'data', 'YCB_Object')
#opt = ycb_video()

# classes = opt.classes + ('holiday_cup1', 'holiday_cup2', 'sanning_mug_new', )
# classes = ('block_red', 'block_green', 'block_blue', 'block_yellow')
classes = ('industrial_dolly',)

# extent file
filename = osp.join(root_path, 'models', 'extents_dolly.txt')
fext = open(filename, "w")

for i in range(len(classes)):

    # clear model
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_pattern(pattern="Camera")
    bpy.ops.object.select_all(action='INVERT')
    bpy.ops.object.delete()

    cls = classes[i]
    print(cls)
    filename = osp.join(root_path, 'models', cls, 'IndustrialDolly.obj')
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
    m = np.mean(vertices, axis=0)
    factor = 1

    for item in bpy.data.objects:
        if item.type == 'MESH':
            for vertex in item.data.vertices:
                rv = vertex.co
                vertex.co[0] = (vertices[i, 0] - m[0]) * factor
                vertex.co[1] = (vertices[i, 1] - m[1]) * factor
                vertex.co[2] = (vertices[i, 2] - m[2]) * factor
                i += 1
       
    # save model
    bpy.data.objects[obj_object.name].select = True
    filename = osp.join(root_path, 'models', cls, 'textured_simple.obj')
    bpy.ops.export_scene.obj(filepath=filename, use_selection=True)

    vertices[:, 0] = (vertices[:, 0] - m[0]) * factor
    vertices[:, 1] = (vertices[:, 1] - m[1]) * factor
    vertices[:, 2] = (vertices[:, 2] - m[2]) * factor

    # write extent
    extent = 2 * np.max(np.absolute(vertices), axis=0)
    print(extent)
    fext.write('%f %f %f\n' % (extent[0], extent[1], extent[2]))

    # write points
    perm = np.random.permutation(np.arange(vertices.shape[0]))
    index = perm[:3000]
    pcloud = vertices[index, :]

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
