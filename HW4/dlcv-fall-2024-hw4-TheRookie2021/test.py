from plyfile import PlyData, PlyElement

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    print(len(vertices))


fetchPly("dataset/train/sparse/0/points3D.ply")

fetchPly("output/round2_train_only_more_iter/point_cloud/iteration_90000/point_cloud.ply")

fetchPly("output/round3_train_only_iter150000/point_cloud/iteration_150000/point_cloud.ply")
fetchPly("output/round5_no_init_ply/point_cloud/iteration_90000/point_cloud.ply")