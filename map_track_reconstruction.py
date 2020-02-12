import csv
import json
import sys
import math

def read_sys_argv():
    img_w = 1920
    img_h = 1080
    if len(sys.argv) == 3:
        return int(sys.argv[1]), int(sys.argv[2])
    else:
        return img_w, img_h

def denormalize_x(p):
    return round(p * max(IMG_W, IMG_H) - 0.5 + IMG_W / 2.0)

def denormalize_y(p):
    return round(p * max(IMG_W, IMG_H) - 0.5 + IMG_H / 2.0)

def read_tracks_csv(trackscsv_path = 'tracks.csv', get_tracks_im=False):
    tracks = {}
    tracks_im = {}
    f = csv.reader(open(trackscsv_path, 'r'), delimiter='\t')
    next(f, None)
    for l in f:
        (im, track_id, feature_id, x, y, scale, r, g, b) = tuple(l)

        if track_id not in tracks:
            tracks[track_id] = []
        tracks[track_id].append((im, feature_id, x, y, scale, int(r), int(g), int(b)))

        if im not in tracks_im:
            tracks_im[im] = []
        tracks_im[im].append((track_id, feature_id, x, y, scale, int(r), int(g), int(b)))

    if not get_tracks_im:
        return tracks
    elif get_tracks_im:
        return tracks, tracks_im

def load_reconjson(reconjson_path='reconstruction.json'):
    return json.loads(open(reconjson_path).read())[0]

#read reconstruction.json
def make_resultcsv(tracks):
    recon = load_reconjson()

    f = open('result.csv', 'w')
    print('img_name', 'track_id', 'feature_id', 'norm_x', 'norm_y', 'scale', 'r', 'g', 'b', 'pcx', 'pcy', 'pcz', 'denorm_x', 'denorm_y', file=f, sep=',')

    count = 0
    for track_id in recon['points']:
        r, g, b = (recon['points'][track_id]['color'])
        for (im, feature_id, x, y, scale, rr, gg, bb) in tracks[track_id]:
            if r == int(r) and gg == int(g) and bb == int(b):
                pcx, pcy, pcz = recon['points'][track_id]['coordinates']
                normed_x = denormalize_x(float(x))
                normed_y = denormalize_y(float(y))
                print(im, track_id, feature_id, x, y, scale, rr, gg, bb, pcx, pcy, pcz, normed_x, normed_y, file=f, sep=',')
                count += 1
                break
    f.close()

if __name__ == '__main__':
    # IMG_W, IMG_H = read_sys_argv()
    # tracks = read_tracks_csv()
    # make_resultcsv(tracks)

    tracks = read_tracks_csv()
    fids = {}
    for track in tracks.values():
        for track_elem in track:
            if track_elem[0] + track_elem[1] in fids:
                print('duplicated')
                break
            else:
                fids[track_elem[0] + track_elem[1]] = True
        





