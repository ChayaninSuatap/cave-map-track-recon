import json
def fn(reconjson_path = 'reconstruction.json'):
    f = open('recon.ply', 'w')
    print('ply\nformat ascii 1.0', file = f)

    recon = json.loads(open(reconjson_path).read())[0]

    points_n = len(list(recon['points'].keys()))

    print('element vertex', points_n, file = f)
    print('''property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header''', file = f)

    for track_id in recon['points'].keys():
        r, g, b = recon['points'][track_id]['color']
        x, y, z = recon['points'][track_id]['coordinates']
        print(x,y,z,int(r),int(g),int(b), file = f)
    
    f.close()

if __name__ == '__main__':
    # fn('undistorted_reconstruction.json')
    fn()

