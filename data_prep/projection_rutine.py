import os
import glob
from datetime import datetime,timedelta
import re
import zipfile

from rasterio.transform import from_origin
from PIL import Image
import numpy as np
from netCDF4 import Dataset
from scipy.spatial import Delaunay

import matplotlib.pyplot as plt
from scipy.integrate import simpson

import logging
logging.basicConfig(level=logging.INFO)



METEOSAT_ROOT_FOLDER = '/media/vladlanda/DATA/Meteosat'
LI_FOLDER = 'afa'
IR_FOLDER = 'ir_105'

LI_ROOT_FOLDER = f'{METEOSAT_ROOT_FOLDER}{os.sep}{LI_FOLDER}'
IR_ROOT_FOLDER = f'{METEOSAT_ROOT_FOLDER}{os.sep}{IR_FOLDER}'

DATE_FOLDER_FORMAT = '%Y_%m_%d'
WLD_DATE_FORMAT = '%Y%m%dT%H%M'
NC_DATE_FORMAT = '%Y%m%d%H%M%S'

WLD_REG_EXPRESSION = r'\d{8}T\d{6}Z{1}_\d{8}T\d{6}Z{1}'
NC_REG_EXPRESSION = r'\d{14}_\d{14}_N'


class Inter2DEfficient(object):

    def __init__(self,source_lonlats,target_lonlats):

        s_lons,s_lats = source_lonlats
        t_lons,t_lats = target_lonlats

        xy = np.vstack((s_lons.ravel(),s_lats.ravel())).T
        uv = np.vstack((t_lons.ravel(), t_lats.ravel())).T

        self.vtx, self.wts = self._interp_weights(xy, uv)

    def _interp_weights(self,xy, uv,d=2):
        tri = Delaunay(xy)
        simplex = tri.find_simplex(uv)
        vertices = np.take(tri.simplices, simplex, axis=0)
        temp = np.take(tri.transform, simplex, axis=0)
        delta = uv - temp[:, d]
        bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
        return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))

    def interpolate(self,values,shape):
        arr = np.einsum('nj,nj->n', np.take(values.ravel(), self.vtx), self.wts)
        arr = arr.reshape(shape)
        return arr


class LInetCDF(object):

    def __init__(self,nc_path):

        self.nc = Dataset(nc_path)
        self.dx = self.dy = 2 # 2km

    def _azel_to_latlon(self,az: np.array ,el:np.array ,r_eq=6378137.0,f=1/298.257223563,h=6378137.0+35786400.0,lambda_d=0):

        lambda_s = az
        f_s      = el
        r_pol = r_eq * (1-f)

        s_4 = r_eq**2 / r_pol**2
        s_5 = h**2 - r_eq**2
        sd = np.sqrt(np.clip((h*np.cos(lambda_s)*np.cos(f_s))**2-(np.cos(f_s)**2 + s_4*np.sin(f_s)**2)*s_5,a_min=0,a_max=None))
        s_n = (h*np.cos(lambda_s)*np.cos(f_s) - sd) / (np.cos(f_s)**2 + s_4*np.sin(f_s)**2)

        s_1 = h-s_n*np.cos(lambda_s)*np.cos(f_s)
        s_2 = -s_n*np.sin(lambda_s)*np.cos(f_s)
        s_3 = s_n*np.sin(f_s)

        s_xy = np.sqrt(s_1**2+s_2**2)

        lon = np.arctan(s_2/s_1)+lambda_d
        lat = np.arctan(s_4*s_3/s_xy)

        lon = np.rad2deg(lon)
        lat = np.rad2deg(lat)

        return lat,lon
    
    def generate_full_LI_coverage_matrix(self):

        _lambda_0 = 1.55561889270898e-01
        _f_0      = -1.55561889270898e-01
        _ags = _egs = 5.58871526031607e-05
        _n_steps = 5568
        
        azimuths   = -(np.linspace(0,(_n_steps - 1) * _ags,_n_steps) - _lambda_0)
        elevations = np.linspace(0,(_n_steps - 1) * _egs,_n_steps) + _f_0

        az2d,el2d = np.meshgrid(azimuths,elevations)

        lat,lon = self._azel_to_latlon(az2d.reshape(-1),el2d.reshape(-1))
        # lat,lon = _azel_to_latlon(azimuths.reshape(-1),elevations.reshape(-1))
        lat = lat.reshape(-1,_n_steps)
        lon = lon.reshape(-1,_n_steps)

        # lat = np.flip(lat,axis=0)
        # lon = np.flip(lon,axis=1)

        rad_lat = np.deg2rad(-lat)
        rad_lon = np.deg2rad(lon)

        return -lat,lon
        # return rad_lat,rad_lon

    def nc_to_image(self):

        nc = self.nc
        
        type_dict = {'AF':'flash_accumulation','AFA':'accumulated_flash_area'}

        lambda_s = np.array(nc.variables['x'][:]) 
        lambda_0 = nc.variables['x'].add_offset
        ags      = nc.variables['x'].scale_factor

        f_s      = np.array(nc.variables['y'][:])
        f_0      = nc.variables['y'].add_offset
        egs      = nc.variables['y'].scale_factor

        c = (lambda_s - lambda_0)/ags #+ 1
        r = -(f_s - f_0)/egs #+ 1

        c = np.rint(c)
        r = np.rint(r)

        li_map = np.zeros((5568,5568))
        index_map = np.zeros_like(li_map)
        
        try:
            accumulated_data = nc.variables[type_dict[nc.type] ][:] * nc.variables[type_dict[nc.type]].scale_factor
        except:
            accumulated_data = nc.variables[type_dict[nc.type]][:]

        indecies = np.array(list(zip(np.array(c,dtype=int),np.array(r,dtype=int))))

        for k,(az,el) in enumerate(indecies):
            # print(az,el)
            index_map[el,az] += 1
            li_map   [el,az] += accumulated_data[k]

        mask = li_map > 0

        if nc.type == 'AF':        
            a = li_map.copy()
            a[mask] = np.log(a[mask]) - np.log(a[mask]).mean() 
            a[mask] /= a[mask].var()
            li_map = a
        if nc.type == 'AFA':
            li_map = li_map / np.max(li_map) * 255
            

        # li_map /= simpson(simpson(li_map,dx=self.dx),dx=self.dy)
        # index_map /= simpson(simpson(index_map,dx=self.dx),dx=self.dy)
        index_map = index_map / np.max(index_map) * 255
        
        return li_map,index_map

class LIProjector(object):

    def __init__(self,jpeg_path=None,wld_path=None,nc_path=None):


        # self.update_files(jpeg_path,wld_path,nc_path)

        # print(self.transform)
        # print(self.wld_data)
        # print(self.image_array.shape)
        pass

    def _read_wld(self,wld_path):

        with open(wld_path, "r") as f:
            wld_data = [float(line.strip()) for line in f.readlines()]
        return wld_data
    
    def _get_transform(self,wld_data):

        pixel_x         = wld_data[0]  # Pixel size in X direction (degrees per pixel)
        pixel_y         = wld_data[3]  # Negative Pixel size in Y direction
        upper_left_x    = wld_data[4]  # Top-left X coordinate (Longitude)
        upper_left_y    = wld_data[5]  # Top-left Y coordinate (Latitude)

        # Define raster transformation
        transform = from_origin(upper_left_x, upper_left_y, pixel_x, -pixel_y)

        return transform

    def _get_meshgrid(self):
        # -----------------------------
        # Step 2: Read the Meteosat JPEG Image and World File (.wld)
        # -----------------------------

        height, width = self.image_array.shape[:2]
        pixel_x         = self.wld_data[0]  # Pixel size in X direction (degrees per pixel)
        pixel_y         = self.wld_data[3]  # Negative Pixel size in Y direction
        upper_left_x    = self.wld_data[4]  # Top-left X coordinate (Longitude)
        upper_left_y    = self.wld_data[5]  # Top-left Y coordinate (Latitude)

        # -----------------------------
        # Step 3: Regrid the Lightning Data to Match the Image Grid
        # -----------------------------
        # Create target grid matching the JPEG resolution
        jpeg_lons = np.linspace(upper_left_x, upper_left_x + width * pixel_x, width)
        jpeg_lats = np.linspace(upper_left_y, upper_left_y + height * pixel_y, height)
        jpeg_lons, jpeg_lats = np.meshgrid(jpeg_lons, jpeg_lats)

        return jpeg_lons, jpeg_lats

    def update_files(self,jpeg_path,wld_path,nc_path):

        self.nc = LInetCDF(nc_path)
        wld = self._read_wld(wld_path)
        new_transform = self._get_transform(wld)

        new_info = False
    
        if not hasattr(self,'transform') or hash(new_transform) != hash(self.transform):

            # TODO update transform
            self.transform = new_transform
            self.wld_data = wld
            # TODO update image
            self.image_array = np.array(Image.open(jpeg_path))
            # TODO build new meshgrid
            logging.info("_get_meshgrid().....")
            self.jpeg_lons, self.jpeg_lats = self._get_meshgrid()

            new_info = True

        if not hasattr(self,'nc_lats') or not hasattr(self,'nc_lons'):
            logging.info("generate_full_LI_coverage_matrix().....")
            self.nc_lats,self.nc_lons = self.nc.generate_full_LI_coverage_matrix()
            new_info = True

        if not hasattr(self,'iterp') or new_info:
            logging.info("Inter2DEfficient().....")
            self.iterp = Inter2DEfficient((self.nc_lons,self.nc_lats),(self.jpeg_lons, self.jpeg_lats))

    def process_projection(self,save_file_name,postfix='LI',index_postfix='COUNT'):

        logging.info("process_projection().....")
        li_map,index_map = self.nc.nc_to_image()

        
        projected_li_map = self.iterp.interpolate(li_map,self.image_array.shape)
        projected_index_map = self.iterp.interpolate(index_map,self.image_array.shape)

        # projected_index_map = projected_index_map / np.max(projected_index_map) * 255

        logging.info(f"Saving file to {save_file_name}...")
        os.makedirs(os.path.dirname(save_file_name),exist_ok=True)

        Image.fromarray(np.asarray(projected_li_map,dtype=np.uint8),mode='L').save(save_file_name) 
        Image.fromarray(np.asarray(projected_index_map,dtype=np.uint8),mode='L').save(save_file_name.replace(postfix,index_postfix))


        return projected_li_map,projected_index_map






        
        





def get_files_triples(source_folder,target_folder):

    '''
    return triple (jpg,wld,LI_zip)
    '''

    source_jpg_files = glob.glob(f'{source_folder}{os.sep}*.jpg')
    source_wld_files = glob.glob(f'{source_folder}{os.sep}*.wld')

    source_wld_files = [wld for wld in source_wld_files if wld.replace('.wld','.jpg') in source_jpg_files]

    missing = [wld for wld in source_wld_files if not wld.replace('.wld','.jpg') in source_jpg_files] + [jpg for jpg in source_jpg_files if not jpg.replace('.jpg','.wld') in source_wld_files]

    if len(missing) > 0: print(missing)
    # print(len(source_jpg_files) , len(source_wld_files))
    assert len(source_jpg_files) == len(source_wld_files)

    source_reg_obj = re.compile(WLD_REG_EXPRESSION)
    source_files_dates = [datetime.strptime(source_reg_obj.findall(wld_file)[0][:13],WLD_DATE_FORMAT) for wld_file in source_wld_files]


    all_target_files = glob.glob(f'{target_folder}{os.sep}*.zip')
    # zip_target_files = [file for date in source_files_dates for file in all_target_files if datetime.strftime(date+timedelta(minutes=10),NC_DATE_FORMAT)+'_N' in file]
    triples = [(jpg,wld,file) for jpg,wld,date in zip(source_jpg_files,source_wld_files,source_files_dates) for file in all_target_files if datetime.strftime(date+timedelta(minutes=10),NC_DATE_FORMAT)+'_N' in file]
    
    if len(triples) <= 0 : return [],[],[]

    source_jpg_files,source_wld_files,zip_target_files = list(zip(*triples))


    assert len(zip_target_files) == len(source_jpg_files)

    return source_jpg_files,source_wld_files,zip_target_files

def unzip_and_rename_li(list_of_zip_files):

    list_of_unziped_files = [zip_file.replace('.zip','.nc') for zip_file in list_of_zip_files]

    # zip_file = list_of_zip_files[0]

    for zip_file,unziped_file in zip(list_of_zip_files,list_of_unziped_files):
        zipped = zipfile.ZipFile(zip_file)
        if os.path.isfile(unziped_file): continue

        # try:
        zipped_nc = [zf for zf in zipped.namelist() if '.nc' in zf and 'BODY' in zf][0]
        # except:
        #     print(f'Error : no BODY in .nc {zipped}')
        #     print(f'List of files in .zf :{zipped.namelist()}')
        #     continue
        nc = zipped.open(zipped_nc)
        with open(unziped_file,'wb') as f:
            f.write(nc.read())

    return list_of_unziped_files

def get_date_folder_pairs():

    source_date_folders = [folder for folder in glob.glob(f'{IR_ROOT_FOLDER}{os.sep}*') if os.path.isdir(folder)]
    source_dates = [datetime.strptime(folder.split(os.sep)[-1],DATE_FOLDER_FORMAT) for folder in source_date_folders]

    target_potential_date_folders = [f'{LI_ROOT_FOLDER}{os.sep}{date.strftime(DATE_FOLDER_FORMAT)}' for date in source_dates]
    source_target_date_folders_pairs = [(source_f,target_f) for source_f,target_f in zip(source_date_folders,target_potential_date_folders) if os.path.isdir(target_f)]

    source_target_date_folders_pairs = [pair for _,pair in sorted(zip(source_dates,source_target_date_folders_pairs))]

    return source_target_date_folders_pairs



def main():

    OVER_WRITE = False

    lip = LIProjector()
    source_target_date_folders_pairs = get_date_folder_pairs()[-1:]

    for sf,tf in source_target_date_folders_pairs:
        
        logging.info(f'Getting files triplets of {sf}...')
        source_jpg_files,source_wld_files,zip_target_files = get_files_triples(sf,tf)
        logging.info(f'Unzipping files in {tf}...')
        unzip_target_files = unzip_and_rename_li(zip_target_files)
        # logging.info(f'Projecting files in {unzip_target_files}...')

        for jpg,wld,unzip in list(zip(source_jpg_files,source_wld_files,unzip_target_files)):

            save_file_name = os.path.dirname(unzip).replace(LI_FOLDER,f'{LI_FOLDER}_projected')
            save_file_name = os.path.join(save_file_name,os.path.basename(jpg.replace('band','LI')))

            if not os.path.isfile(save_file_name) or OVER_WRITE:

                lip.update_files(jpg,wld,unzip)
                try:
                    projected_li_map,projected_index_map = lip.process_projection(save_file_name)
                except:
                    print(f'Error : {save_file_name}')

if __name__ == '__main__':

    main()

    # tests()
    # source_target_date_folders_pairs = get_date_folder_pairs()
    # print(source_target_date_folders_pairs[-1:])

    # f2 = '/media/vladlanda/DATA/Meteosat/ir_105/2024_10_01/FCIL1FDHSI_20241001T123007Z_20241001T123924Z_epct_a63051a8_FP_band40.jpg'
    # # f2 = '/media/vladlanda/DATA/Meteosat/ir_105/2024_10_01/FCIL1FDHSI_20241001T122007Z_20241001T122924Z_epct_a7db959b_FP_band40.jpg'
    # im2 = Image.open(f2)
    # plt.imshow(im2,cmap='gray')

    # # f = '/media/vladlanda/DATA/Meteosat/afa_projected/2024_10_01/FCIL1FDHSI_20241001T123007Z_20241001T123924Z_epct_a63051a8_FP_LI40.jpg'
    # f = '/media/vladlanda/DATA/Meteosat/afa_projected/2024_10_01/FCIL1FDHSI_20241001T122007Z_20241001T122924Z_epct_a7db959b_FP_LI40.jpg'
    # im = Image.open(f)
    # im = np.array(im,dtype=np.float32)
    # im[im <= 0] = np.nan 
    # plt.imshow(im,alpha=.4)
    # plt.colorbar()
    # plt.show()


    # a,b,c = list(zip(*[(1, 2 , 11), (3, 4 , 12), (5, 6 , 13)]))

    # print(a)
    # print(b)
    # print(c)