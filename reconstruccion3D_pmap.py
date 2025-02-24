import os
import pathlib
import pycolmap
import open3d as o3d

image_dir = pathlib.Path('cubo2')
output_dir = pathlib.Path('colmap_data')

output_dir.mkdir(exist_ok=True)

mvs_path = output_dir / "mvs"
database_path = output_dir / "database.db"

# Run the feature extraction
pycolmap.extract_features(database_path, image_dir)

# Run the exhaustive matcher
pycolmap.match_exhaustive(database_path)

# Run the incremental mapping
maps = pycolmap.incremental_mapping(database_path, image_dir, output_dir)
maps[0].write(output_dir)

# Dense reconstruction without CUDA
pycolmap.undistort_images(mvs_path, output_dir, image_dir)
#pycolmap.patch_match_stereo(mvs_path, use_cuda=False)  # Disable CUDA
pycolmap.stereo_fusion(mvs_path / "dense.ply", mvs_path)

# Obtain the point cloud
ply_path = mvs_path / "dense.ply"

if not ply_path.exists():
    print('No se encontró el archivo PLY')
    exit()

# Load the point cloud
nube_puntos = o3d.io.read_point_cloud(str(ply_path))

# Visualize the point cloud
o3d.visualization.draw_geometries([nube_puntos])

# Example of setting SIFT options
sift_options = pycolmap.SiftExtractionOptions()
sift_options.max_num_features = 512
pycolmap.extract_features(database_path, image_dir, sift_options=sift_options)