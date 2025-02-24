import os
import pathlib
import pycolmap
import open3d as o3d

image_dir = pathlib.Path('cubo2')
output_dir = pathlib.Path('colmap_data')

output_dir.mkdir(exist_ok=True)

mvs_path = output_dir / "mvs"
database_path = output_dir / "database.db"

#Extracción de caracteristicas
pycolmap.extract_features(database_path, image_dir)

pycolmap.match_exhaustive(database_path)

maps = pycolmap.incremental_mapping(database_path, image_dir, output_dir)
maps[0].write(output_dir)

#Reconstrucción densa
pycolmap.undistort_images(mvs_path, output_dir, image_dir)
#pycolmap.patch_match_stereo(mvs_path, use_cuda=False)  # Sin CUDA porque no tenga GPU NVIDIA
pycolmap.stereo_fusion(mvs_path / "dense.ply", mvs_path)

#Obtener la nube de puntos
ply_path = mvs_path / "dense.ply"

if not ply_path.exists():
    print('No se encontró el archivo PLY')
    exit()

#Cargar la nube de puntos
nube_puntos = o3d.io.read_point_cloud(str(ply_path))

#Visualizar la nube de puntos
o3d.visualization.draw_geometries([nube_puntos])

#SIFT
sift_options = pycolmap.SiftExtractionOptions()
sift_options.max_num_features = 512
pycolmap.extract_features(database_path, image_dir, sift_options=sift_options)
