from numpy import load
from map_track_reconstruction import read_sys_argv, read_tracks_csv
import matplotlib.pyplot as plt

def denormalize_x(p, IMG_W, IMG_H):
    return round(p * max(IMG_W, IMG_H) - 0.5 + IMG_W / 2.0)

def denormalize_y(p, IMG_W, IMG_H):
    return round(p * max(IMG_W, IMG_H) - 0.5 + IMG_H / 2.0)

def get_xy_from_features(fn, denormalize=True):
    o = load(fn)
    l = []
    
    for x in o['points']:
        x, y, size, angle = x
        if denormalize:
            x = denormalize_x(x)
            y = denormalize_y(y)
        l.append((x, y))
    return l

def get_fids_from_tracks(img_fn, tracks):
    output = []
    for track in tracks.values():
        for track_elem in track:
            if img_fn == track_elem[0]:
                output.append( track_elem[1])
    return output

def get_fids_from_tracks_im(img_fn, tracks_im):
    output = []
    for track_elem in tracks_im[img_fn]:
        output.append( track_elem[1])
    return output

if __name__ == '__main__':
    pass
    # fids = get_fids_from_tracks(img_fn = '001.jpg')
    # print(fids)

    # IMG_W, IMG_H = read_sys_argv()
    # points = get_xy_from_features('gopro_board_game_data/features/000.jpg.features.npz')

    # img = plt.imread('gopro_board_game_data/images/000.jpg')
    # fig, ax = plt.subplots()
    # ax.imshow(img)

    # xs = []
    # ys = []
    # for i,(x,y) in enumerate(points):
    #     if  str(i) in fids:
    #         xs.append(x)
    #         ys.append(y)

    # ax.plot(xs,ys, 'bo', markersize=5)
    # print(len(points), len(xs))
    # plt.show()



