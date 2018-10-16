import os.path as osp

this_dir = osp.dirname(__file__)
root_path = osp.join(this_dir, '..', 'data', 'YCB_Video')

# read train index
filename = osp.join(root_path, 'train.txt')
video_ids = set([])
with open(filename) as f:
    for x in f.readlines():
        index = x.rstrip('\n')
        pos = index.find('/')
        video_id = index[:pos]
        video_ids.add(video_id)

# test index
filename = osp.join(root_path, 'debug.txt')
ftest = open(filename, 'w')
video_ids = list(video_ids)
video_ids.sort()
for i in range(len(video_ids)):
    line = '%s/000001\n' % (video_ids[i])
    ftest.write(line)
