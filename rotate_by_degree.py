from plyfile import PlyData, PlyElement
import sys
from auto_rotate import make_rotator, gen_ply_vectrices, apply_rotation, write_rotated_ply

if __name__ == '__main__':
    path = sys.argv[1]
    degree = float(sys.argv[2])
    rotator = make_rotator(degree * 0.0174533)
    rotated_vertrices = []
    input_ply = PlyData.read(path)

    #apply rotation
    for t in gen_ply_vectrices(input_ply):
        x,y,z = t[0], t[1], t[2]
        newx, newy, newz = apply_rotation(x,y,z, rotator)
        rotated_vertrices.append((newx, newy, newz))

    #make new ply
    write_rotated_ply(rotated_vertrices, input_ply, 'rotated_by_degree.ply')