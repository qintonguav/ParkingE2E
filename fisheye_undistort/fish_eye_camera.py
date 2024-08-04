import numpy as np
import os
import tqdm

class FisheyeCamera:
    def __init__(self, para_path) -> None:
        self.para_path = para_path


    def check_undistort_info(self, camera_tag, sf = 4.0):
        """
        Inputs:
        camera_tag: camera channel tag
        sf: Affect the result of Scaramuzza's undistortion. Try to change it to see how it affects the result
        Outputs:
        mapx_persp_32: mapping in x direction
        mapy_persp_32: mappint in y direction
        """
        mapx_persp_32_path = os.path.join(self.para_path, camera_tag, "mapx_persp_32_{}.npy".format(camera_tag))
        mapy_persp_32_path = os.path.join(self.para_path, camera_tag, "mapy_persp_32_{}.npy".format(camera_tag))
        if os.path.exists(mapx_persp_32_path) and os.path.exists(mapy_persp_32_path):
            mapx_persp_32 = np.load(mapx_persp_32_path)
            mapy_persp_32 = np.load(mapy_persp_32_path)
        else:
            print("Save {} camera mapping, waiting...".format(camera_tag))
            path_ocam = os.path.join(self.para_path, camera_tag, "calib_results_{}.txt".format(camera_tag))
            o = self.get_ocam_model(path_ocam)
            mapx_persp, mapy_persp = self.create_perspective_undistortion_LUT(o, sf)
            mapx_persp_32 = mapx_persp.astype('float32')
            mapy_persp_32 = mapy_persp.astype('float32')
            np.save(mapx_persp_32_path, mapx_persp_32)
            np.save(mapy_persp_32_path, mapy_persp_32)

        return mapx_persp_32, mapy_persp_32


    def get_ocam_model(self, filename):
        o = {}
        with open(filename) as f:
            lines = [l for l in f]
            
            l = lines[2]
            data = l.split()
            o['length_pol'] = int(data[0])
            o['pol'] = [float(d) for d in data[1:]]
            
            l = lines[6]
            data = l.split()
            o['length_invpol'] = int(data[0])
            o['invpol'] = [float(d) for d in data[1:]]
            
            l = lines[10]
            data = l.split()
            o['xc'] = float(data[0])
            o['yc'] = float(data[1])
            
            l = lines[14]
            data = l.split()
            o['c'] = float(data[0])
            o['d'] = float(data[1])
            o['e'] = float(data[2])
                    
            l = lines[18]
            data = l.split()
            o['height'] = int(data[0])
            o['width'] = int(data[1])

        return o

    def create_perspective_undistortion_LUT(self, o, sf):

        mapx = np.zeros((o['height'],o['width']))    
        mapy = np.zeros((o['height'],o['width']))    
        
        Nxc = o['height']/2.0
        Nyc = o['width']/2.0   
        Nz = -o['width']/sf       

        for i in tqdm.tqdm(range(o['height'])):
            for j in range(o['width']):
                M = []
                M.append(i - Nxc)
                M.append(j - Nyc)
                M.append(Nz)
                m = self.world2cam(M, o)     
                mapx[i,j] = m[1]
                mapy[i,j] = m[0]
                
        return mapx, mapy


    def world2cam(self, point3D, o):
        point2D = []    
        
        norm = np.linalg.norm(point3D[:2])

        if norm != 0:
            theta = np.arctan(point3D[2]/norm)
            invnorm = 1.0/norm
            t = theta
            rho = o['invpol'][0]
            t_i = 1.0
            
            for i in range(1,o['length_invpol']):
                t_i *= t
                rho += t_i*o['invpol'][i]
                
            x = point3D[0]*invnorm*rho
            y = point3D[1]*invnorm*rho
            
            point2D.append(x*o['c']+y*o['d']+o['xc'])
            point2D.append(x*o['e']+y+o['yc'])
        else:
            point2D.append(o['xc'])
            point2D.append(o['yc'])
            
        return point2D