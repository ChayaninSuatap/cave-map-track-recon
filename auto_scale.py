from map_track_reconstruction import read_tracks_csv, load_reconjson
from read_feature import get_xy_from_features, get_fids_from_tracks, denormalize_x, denormalize_y
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

def compute_distance(x1, y1, z1, x2, y2, z2):
    return ((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)**(1/2)

def compute_distance_2d(x1, y1, x2, y2):
    return ((x1-x2)**2 + (y1-y2)**2)**(1/2)

def compute_average_coord(img_fn, recon, tracks_csv):
    match_track_ids = []
    tracks_in_recon = {k:0 for k in recon['points'].keys()}
    for track in tracks_csv.keys():
        if track in tracks_in_recon:
            for item in tracks_csv[track]:
                if item[0] == img_fn:
                    match_track_ids.append(track)
    
    coords = 0, 0, 0
    count = 0
    for track_id in match_track_ids:
        x,y,z = get_coord_from_reconjson_by_trackid(recon, track_id)
        coords = coords[0] + x, coords[1] + y, coords[2] + z
        count += 1

    return coords[0] / count , coords[1] / count , coords[2] / count
    
        
if __name__ == '__main__':
    # path = 'gopro_board_game_data/odm/'
    path = sys.argv[1] + '/'
    real_distance = float(sys.argv[2])
    #pre
    records = []
    recon = load_reconjson(path + 'opensfm/reconstruction.json')
    tracks_csv = read_tracks_csv(path + 'opensfm/tracks.csv')

    #get all center point of images
    #(x, y, z)

    for image_fn in os.listdir(path + 'images/'):
        print('scanning', image_fn)

        result = get_coords_from_selected_center_image(path, image_fn, recon, tracks_csv)
        if result == False:
            continue

        pcx, pcy, pcz, feature_x, feature_y = result
        rotate_rad = float(image_fn.split('_')[1][:-4])
        
        record = ( image_fn, pcx, pcy, pcz, rotate_rad, feature_x, feature_y)
        records.append( record)

    #compute distance pairs
    pairs = [] #(i,j,distance)
    for i in range(len(records)):
        for j in range(i+1, len(records)):
           _, x1, y1, z1, rad1, _, _ = records[i]
           _, x2, y2, z2, rad2, _, _ = records[j] 

           #old
           #no z
        #    distance = ((x1-x2)**2 + (y1-y2)**2)**(1/2)
           #with z
        #    distance = ((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)**(1/2)

           #new
           rad1 = 2*pi + rad1 if rad1 < 0 else rad1
           rad2 = 2*pi + rad2 if rad2 < 0 else rad2
           diff = fabs(rad1 - rad2)
           distance = 2*pi - diff if diff > pi else diff

           pairs.append( (i, j, distance))
    
    #find best pairs ( max distance)
    best_pair = max(pairs, key = lambda x : x[2])
    img_fn1 = records[best_pair[0]][0]
    img_fn2 = records[best_pair[1]][0]
    print('img fns', img_fn1, img_fn2)

    def print_denormed_feature(record):
        feature_x = record[5]
        feature_y = record[6]
        print(denormalize_x(feature_x, 1920, 1080), denormalize_y(feature_y, 1920, 1080))

    print(records[best_pair[0]])
    print_denormed_feature(records[best_pair[0]])
    print(records[best_pair[1]])
    print_denormed_feature(records[best_pair[1]])
    
    #compute average point
    x1,y1,z1 = compute_average_coord(img_fn1, recon, tracks_csv)
    x2,y2,z2 = compute_average_coord(img_fn2, recon, tracks_csv)

    distance = compute_distance_2d(x1, y1, x2, y2)
    scale_factor = real_distance / distance
    print('ply distance',distance)

    #convert obj to ply
    obj2ply.convert.obj2ply(path + 'odm_texturing/odm_textured_model.obj', path + 'odm_texturing/odm_textured_model.ply')
    input_ply = PlyData.read(path + 'odm_texturing/odm_textured_model.ply')

    #apply scale factor
    print('scaling')
    new_vertrices = []
    for (x,y,z) in gen_ply_vectrices(input_ply):
        newx, newy, newz = x * scale_factor, y * scale_factor, z * scale_factor
        new_vertrices.append((newx, newy, newz))

    #make new ply
    print('writing')
    write_rotated_ply(new_vertrices, input_ply, path + 'odm_texturing/final_scale.ply')

    print('done')
    