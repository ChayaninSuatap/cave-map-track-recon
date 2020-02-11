from map_track_reconstruction import read_tracks_csv, load_reconjson
from read_feature import get_xy_from_features, get_fids_from_tracks
from math import atan2, pi, fabs
from scipy.spatial.transform import Rotation as R
from plyfile import PlyData, PlyElement
import numpy as np
import sys
sys.path.append('obj2ply')
import obj2ply.convert
import os

def get_nearest_center_trackid_xy(path, img_fn, recon_trackids, tracks_csv):
    #open track.csv
    tracks = tracks_csv
    #open feature file
    points = get_xy_from_features(path + 'opensfm/features/%s.features.npz' % (img_fn,), denormalize=False)
    #filter feature point that not in track
    valid_fids = get_fids_from_tracks(img_fn, tracks)
    points = [(i,point) for i,point in enumerate(points) if str(i) in valid_fids]
    #find nearest center point
    def compare_fn(e):
        _,(x,y) = e
        return (x**2 + y**2)**(1/2)

    sorted_points = sorted(points, key=compare_fn)
    #find track id from fid and img_fn
    for (fid,(x,y)) in sorted_points:
        for track_id in tracks.keys():
            if track_id not in recon_trackids:
                continue
            for track_elem in tracks[track_id]:
                if track_elem[0] == img_fn and track_elem[1] == str(fid): #test img_fn and fid
                    return (track_id, x, y)
    #cant find
    return ('sorry', 'cant', 'find')

def get_coord_from_reconjson_by_trackid(recon, trackid):
    x, y, z = recon['points'][str(trackid)]['coordinates']
    return float(x), float(y), float(z)

def get_coords_from_selected_center_image(path, img_fn, recon, tracks_csv):

    track_id, feature_x, feature_y = get_nearest_center_trackid_xy(
        path,
        img_fn,
        recon['points'].keys(),
        tracks_csv)
    
    if (track_id, feature_x, feature_y) == ('sorry', 'cant', 'find'):
        return False

    # #get point that should be at center ( y = 0)
    x, y, z = get_coord_from_reconjson_by_trackid(recon, track_id)
    return x, y, z, feature_x, feature_y

def compute_rotate_rad(y,x, anchor_rad):
    return anchor_rad - atan2(y,x)

def make_rotator(rotate_rad):
    return R.from_euler('xyz', [0, 0, rotate_rad])

def apply_rotation(x,y,z, rotator):
    return rotator.apply([x,y,z])

def gen_ply_vectrices(ply):
    for x in ply['vertex']:
        yield x

def write_rotated_ply(vs, input_ply, output_fn):
    # vs_np = np.array(vs, dtype=[('x','float'),('y','float'),('z','float'), ('red','u1'), ('green', 'u1'), ('blue', 'u1')])
    vs_np = np.array(vs, dtype=[('x','float'),('y','float'),('z','float')])
    PlyData([PlyElement.describe(vs_np, 'vertex'), input_ply['face']], text=True, comments=input_ply.comments).write(output_fn)
        
if __name__ == '__main__':
    # path = 'gopro_board_game_data/odm/'
    path = sys.argv[1] + '/'
    
    #find image that has nearest point to y=0
    #(distance, rotate_rad, x, y, z)
    records = []
    recon = load_reconjson(path + 'opensfm/reconstruction.json')
    tracks_csv = read_tracks_csv(path + 'opensfm/tracks.csv')
    for image_fn in os.listdir(path + 'images/'):
        print('scanning', image_fn)

        result = get_coords_from_selected_center_image(path, image_fn, recon, tracks_csv)
        if result == False:
            continue

        pcx, pcy, pcz, feature_x, feature_y = result
        rotate_rad = float(image_fn.split('_')[1][:-4])
        
        record = (fabs(feature_y), image_fn, rotate_rad, pcx, pcy, pcz)
        records.append( record)
    best_record = min(records, key = lambda x: x[0])
    _,img_name, anchor_rad, pcx, pcy, pcz = best_record
    print('img name', img_name)
    print('will rotate to angle', anchor_rad * 57.2958)
    ### 
    # selected_img_fn = input('select most vertical image : ')
    # pcx, pcy, pcz, _, _ = get_coords_from_selected_center_image(path, selected_img_fn)
    rotate_rad = compute_rotate_rad(pcy, pcx, anchor_rad)
    print('must rotate by', rotate_rad * 57.2958)
    rotator = make_rotator(rotate_rad)
    rotated_vertrices = []

    #convert obj to ply
    obj2ply.convert.obj2ply(path + 'odm_texturing/odm_textured_model.obj', path + 'odm_texturing/odm_textured_model.ply')
    input_ply = PlyData.read(path + 'odm_texturing/odm_textured_model.ply')

    #apply rotation
    for (x,y,z) in gen_ply_vectrices(input_ply):
        newx, newy, newz = apply_rotation(x,y,z, rotator)
        rotated_vertrices.append((newx, newy, newz))

    #make new ply
    write_rotated_ply(rotated_vertrices, input_ply, path + 'odm_texturing/final_rotate.ply')
    