"""These commands will measure intensity and distances between surfaces in ChimeraX"""
#pylint: disable=redefined-builtin
#pylint: disable=expression-not-assigned
#pylint: disable=line-too-long
#pylint: disable=unused-argument
#pylint: disable=too-many-arguments

from chimerax.color_key import show_key
from chimerax.core import colors
from chimerax.std_commands.wait import wait
from chimerax.surface import vertex_convexity
from chimerax.atomic import AtomsArg
from chimerax.core.commands import (BoolArg, Bounded, CmdDesc, ColormapArg,
                                    ColormapRangeArg, Int2Arg, IntArg,
                                    SurfacesArg, StringArg, FloatArg, SurfaceArg, AxisArg)
from chimerax.core.commands.cli import EnumOf
from chimerax.map.volumecommand import volume
from chimerax.std_commands.cd import (cd)
from os.path import exists
import numpy
from chimerax.surface.dust import largest_blobs_triangle_mask 
from numpy import (arccos, array, full, inf, isnan, mean, round, nan, nanmax, nanmean,
                   nanmin, pi, ravel_multi_index, sign, split, sqrt, subtract,
                   count_nonzero, swapaxes, savetxt, column_stack,nansum, nanstd,
                   unique, column_stack, round_, int64, abs, digitize, linspace,
                   zeros, where, delete, shape, ravel, min, shape, isin,flip,
                   ones,asarray)
from scipy.ndimage import (binary_dilation, binary_erosion,
                           generate_binary_structure, iterate_structure, gaussian_filter, gaussian_laplace)
from scipy.spatial import KDTree

from skimage.morphology import (skeletonize,label)
from skimage.feature import canny

from chimerax.map_data import ArrayGridData 
from chimerax.map import volume_from_grid_data

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def distance_series(session, surface, to_surface, knn=5, palette=None, color_range=None, key=False):
    """Wrap the distance measurement for list of surfaces."""
    [measure_distance(surface, to_surface, knn)
     for surface, to_surface in zip(surface, to_surface)]
    recolor_surfaces(session, surface, 'distance', palette, color_range, key)


def intensity_series(session, surface, to_map, radius=15, palette=None, color_range=None,
                      key=False, xnorm=None, ynorm=None, znorm=None, blob = 1, output='none'):
    """Wrap the intensity measurement for list of surfaces."""
    [measure_intensity(session, surface, to_map, radius, xnorm, ynorm, znorm, blob,  output)
    for surface, to_map in zip(surface, to_map)]
    recolor_surfaces(session, surface, 'intensity', palette, color_range, key)


def composite_series(session, surface, green_map, magenta_map, radius=15, palette='green_magenta', green_range=None, magenta_range=None):
    """Wrap the composite measurement for list of surfaces."""
    [measure_composite(surface, green_map, magenta_map, radius)
        for surface, green_map, magenta_map in zip(surface, green_map, magenta_map)]
    recolor_composites(session, surface, palette, green_range, magenta_range)


def topology_series(session, surface, to_cell, radius= 8, target = 'sRBC',
                     size=(.1028,.1028,.1028), palette=None, color_range= 'full', key=False,
                     phi_lim= 90, output = 'None'):
    """this is ment to output a color mapped for topology metrics (phi, theta and distance from the target centroid) This is for the whole timeseries move on to the individual outputs"""
    volume(session, voxel_size= size)
    wait(session,frames=1)
    [measure_topology(session, surface, to_cell, radius, target, size, phi_lim, output)
        for surface, to_cell in zip(surface, to_cell)]
    recolor_surfaces(session, surface,'rpd', palette, color_range, key)

def ridge_series(session, surface, to_surface, to_cell, radius = 8, size = (.1028,.1028,.1028), smoothing_iterations = 20,
                 thresh = 0.3, knn=10, palette = None, color_range='full', key= False, clip = 0.5, output= 'None', track = False,
                 exclusion= 0.5):
    
    """This is designed to identify and track ridges that occur at phagosomes - YML"""

    volume(session, voxel_size= size)
    wait(session, frames=1)

    [measure_ridges(session, surface, to_surface, to_cell, radius, smoothing_iterations, thresh, knn,
                    size, clip, output, track, exclusion)
        for surface, to_surface, to_cell in zip(surface, to_surface, to_cell)]
    plt.close('all')

    recolor_surfaces(session, surface, metric= 'edges', palette=None, color_range='full', key=False)

def voids_seires(session, surface, bg = 110, sd = 0.1, stg = 0.8, dups= None, per = 0.88, drop = 3 ):

    """This function is designed to use the surface models specify to generate new volumes that correspond to internal 
    vesicles in the specified surface model. -YML"""
    find_voids(session, surface, bg, sd, stg, dups, per, drop )

def void_size_series(session, surface, track, tp, output):
    """centers to volumes
    author yml 20240327"""
    void_size(session, surface, track, tp, output)

def Void_Motion_series(session,track,t,output):
    """Length of track
    author yml 20240327"""
    Void_Motion(session,track,t,output)

def recolor_surfaces(session, surface, metric='intensity', palette=None, color_range=None, key=False):
    """Wraps recolor_surface in a list comprehension"""
    keys = full(len(surface), False)
    keys[0] = key
    [recolor_surface(session, surface, metric, palette, color_range, key)
     for surface, key in zip(surface, keys)]


def recolor_composites(session, surface, palette='green_magenta', green_range=None, magenta_range=None, palette_range=(40, 240)):
    """Wraps composite_color in a list comprehension"""
    [composite_color(session, surface, palette, green_range, magenta_range, palette_range)
     for surface in surface]


def measure_distance(surface, to_surface, knn):
    """Measure the local motion of two surfaces."""
    distance, _ = query_tree(surface.vertices, to_surface.vertices, knn=knn)
    surface.distance = distance


def measure_topology(session, surface, to_cell, radius=8, target='sRBC', size=[0.1028,0.1028,0.1028],
                      phi_lim= 90, output= 'None'):
    """This command is designed to output a csv file of the surface metrics:
    areal surface roughness, areal surface roughness standard deviation and surface area per frame.
    Additionally this command can color vertices based on their distance from the target centroid.
    Author: Yoseph Loyd
    Date:20230614"""
    
    """Tell the system what target you are computing the areal roughness of."""
    if target == 'sRBC':
        target_r = 2
    elif target =='mRBC':
        target_r = 2.7
    else:
        return
    #Target not recognized

    """Define the target centroid from mid-range x, y and z coordinates."""
    centroid = mean(to_cell.vertices, axis=0)

    """Vertice x,y and z distances from centroid"""
    x_coord, y_coord, z_coord = split(subtract(surface.vertices, centroid), 3, 1)

    x_coord = x_coord.flatten()
    y_coord = y_coord.flatten()
    z_coord = z_coord.flatten()
    """Converting the cartesian system into spherical coordinates"""
    z_squared = z_coord ** 2
    y_squared = y_coord ** 2
    x_squared = x_coord ** 2
    
    distance = sqrt(z_squared + y_squared + x_squared)
    distxy = sqrt(x_squared + y_squared)
    theta = sign(y_coord)*arccos(x_coord / distxy)
    phi = arccos(z_coord / distance) * (180/pi)

    """Logic to identify vertices in the targets local (defined by radius input) around target's upper hemisphere"""
    abovePhi = phi <= phi_lim
    outerlim = (distance  < radius)
    radialClose = outerlim & (distance > target_r)


    """Logic statements for solving the unique X,Y coordinates in the upper hemisphere search"""
    XYZ_SearchLim = distance*abovePhi
    SearchR = (distance*abovePhi*radialClose)>0
    '''XY_SearchR = distance*abovePhi
    XY_deletes = where(XY_SearchR==0)
    XYZ_deletes = where(XYZ_SearchR==0)



    """Solving for unique X,Y,Z coordinates in the upper hemisphere search"""
    xx=x_coord*Points
    yy=y_coord*Points
    zz=z_coord*Points

    xy_raw = column_stack((xx,yy))
    xyz_raw = column_stack((xx,yy,zz))

    xy=unique(delete(xy_raw,XY_deletes,axis=0),axis=0)
    xyz=unique(delete(xyz_raw,XYZ_deletes,axis=0),axis=0)

    """Defining the pixel size from human defined parameter"""
    width = size[0]

    """Defining steps that will are approximately one pixel in length"""
    steps = int64(round_(abs((2*radius)/(width))))

    """Indexing the vertices that fall in one pixel of eachother along each axis""" 
    """Weird nearest neighbors approach"""
    xbins_xy = digitize(xyz[:,0],linspace(-8,8,steps))
    ybins_xy = digitize(xyz[:,1],linspace(-8,8,steps))

    xbins = digitize(xyz[:,0],linspace(-8,8,steps))
    ybins = digitize(xyz[:,1],linspace(-8,8,steps))
    zbins = digitize(xyz[:,2],linspace(-8,8,steps))

    """Making an artificial binary mask of binned vertices into 'pixels' from vertice location"""
    ArtImgxy= zeros([steps,steps])
    ArtImgxy[xbins_xy,ybins_xy]= 1

    ArtImgxyz= zeros([steps,steps,steps])
    ArtImgxyz[xbins,ybins,zbins]= 1

    """Filling holes and cutts in image"""
    ArtImg_Filled= binary_erosion(((gaussian_filter(ArtImgxy,.5))>0),border_value=1,iterations=2)

    ArtImg_Filledxyz = binary_erosion(((gaussian_filter(ArtImgxyz,.2))>0),border_value=1,iterations=1)'''
    ArtImg = ImgReconstruct(SearchR,x_coord,y_coord,z_coord,XYZ_SearchLim,radius=radius, size=size)
    """Area of pixels in X,Y plane of the hemisphere search"""
    Area_S= count_nonzero(ArtImg) * (size[1] * size[0])

    """Outputs for coloring vertices as surface. arguments"""
    radialDistanceAbovePhiLimitxy = abovePhi * radialClose * distance
    surface.radialDistanceAbovePhiNoNans= abovePhi * radialClose * distance 
    radialDistanceAbovePhiLimitxy[radialDistanceAbovePhiLimitxy == 0] = nan

    surface.radialDistanceAbovePhi= abovePhi* distance
    surface.radialDistanceAbovePhiLimitxy=radialDistanceAbovePhiLimitxy

    surface.radialDistance = distance
    surface.theta = theta
    surface.phi = phi
    surface.areasearch = SearchR

    """Single value outputs for definning topology"""
    surface.IRDFCarray = nanmean(radialDistanceAbovePhiLimitxy)
    surface.Sum = nansum(radialDistanceAbovePhiLimitxy)

    surface.area = count_nonzero(ArtImg) * size[0]*size[1]
    surface.ArealRoughness = sqrt(surface.IRDFCarray**2/(2*pi*target_r**2))
    surface.ArealRoughness_STD = nanstd(surface.radialDistanceAbovePhiLimitxy)/(2*pi*target_r**2)
    surface.ArealRoughnessperArea= surface.ArealRoughness / Area_S

    """Text file output"""
    path = exists(output)
    if path == True:
        cd(session,str(output))
        with open('Areal Surface Roughness.csv', 'ab') as f:
            savetxt(f, column_stack([surface.ArealRoughness, surface.ArealRoughness_STD, surface.area, surface.ArealRoughnessperArea]),
                     header=f"Areal-Surface-Roughness-S_q STD_Areal-Rougheness Surface_Area ArealRoughness/um^2", comments='')
    else:
        return surface.radialDistanceAbovePhiNoNans

def measure_ridges(session, surface, to_surface, to_cell,  radius = 8, smoothing_iterations = 20,
                    thresh = 0.3, knn=10, size=[0.1028,0.1028,0.1028], clip=0.5, output= 'None', track = False,
                    exclusion = 0.5):
    
    """Define the target centroid from mid range x,y and z coordinates."""
    centroid = mean(to_cell.vertices, axis=0)

    """Vertice x,y and z coordinates from centroid"""
    x_coord, y_coord, z_coord = split(subtract(surface.vertices, centroid), 3, 1)

    """Defining coordinates for surface t"""
    x_coord = x_coord.flatten()
    y_coord = y_coord.flatten()
    z_coord = z_coord.flatten()

    """Converting the cartesian coordinates into spherical coordinates for surface t"""
    z_squared = z_coord ** 2
    y_squared = y_coord ** 2
    x_squared = x_coord ** 2
    
    distance = sqrt(z_squared + y_squared + x_squared)
    distxy = sqrt(x_squared + y_squared)
    theta = sign(y_coord)*arccos(x_coord / distxy)
    phi = arccos(z_coord / distance) * (180/pi)

    """Paletting options for R, phi, and theta for surface t"""
    surface.radialDistance = distance
    surface.theta = theta
    surface.phi = phi
    
    """Defining search restrictions"""
    sphere = distance  < radius

    try: Clip= z_coord > (min( z_coord[ where(phi<165) ])+clip)
    except ValueError:  #raised if `Clip` is empty.
        pass

    """Defining surfaces t and t+1 convexity without paletting"""
    con = vertex_convexity(surface.vertices, surface.triangles, smoothing_iterations)
    
    ind = (con > thresh)

    car = zeros(shape(surface.vertices))

    car[:,0]= surface.vertices[:,0]*ind*sphere
    car[:,1]= surface.vertices[:,1]*ind*sphere
    car[:,2]= surface.vertices[:,2]*ind*sphere
    
    """Solving for the surface area of high curved regions and the high curve path length"""

    """High curve edges"""
    surface.edges = ind * sphere *Clip + 0

    """search limitations """
    SearchLimit = sphere * Clip

    """Reconstructed image"""
    ArtImg_edges = ImgReconstruct(surface.edges, x_coord, y_coord, z_coord, SearchLim=surface.edges, radius=radius, size=size)
    ArtImg_whole = ImgReconstruct(1 ,x_coord, y_coord, z_coord, SearchLim= SearchLimit, radius=radius, size=size)
    
    """Equivalent to surface dusting all membranes disconnected from the largest object"""
    ArtCC = label(ArtImg_whole)
    cc_num, countscc= unique(ArtCC,return_counts=True)
    fcc = countscc
    fcc[fcc==max(fcc)] = 0
    dusted_value = cc_num[where(fcc==max(fcc))]
    dusted = ( ArtCC == dusted_value) + 0

    ArtImg = ArtImg_edges*dusted

    """Surface Area of high curved regions"""
    surface.Area = count_nonzero(ArtImg) * size[0]*size[1]

    """Skeletonizing the reconstructed image"""
    RidgePathLength = skeletonize((ArtImg*1),method='lee')
    """Connected components: paths"""
    RidgeCc = label(RidgePathLength)
    _,counts=unique(RidgeCc,return_counts=True)
    """excluding all zero points in the matrix / Image"""
    f=counts
    f[f==max(counts)]=0
    """size exclusion"""
    size_exclusion= exclusion / size[0]
    p=where(f >= size_exclusion)

    surface.pathlengthsabovethresh= count_nonzero(isin(RidgeCc,p)==True)*size[0]

    surface.pathlength = count_nonzero(RidgePathLength) * size[0]
    surface.pathlengthsabovethresh= count_nonzero(isin(RidgeCc,p)==True)*size[0]

    """Text file output"""
    path = exists(output)
    if path == True:
        cd(session,str(output))
        with open('RidgeInfo.csv', 'ab') as f:
            savetxt(f, column_stack([surface.Area, surface.pathlength, surface.pathlengthsabovethresh]),
                     header=f"High_Curve_Surface_Area Lamella_pathlength Lamella_pathlength_above_thresh", comments='')
        pos=where(isin(RidgeCc,p)==True)
        frame=str(surface.id[1])
        fig= plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        ax.scatter(pos[0], pos[1], pos[2], c='black')
        ax.set_title('Skeletonized Edges')
        ax.set_xlabel('X \u03BCm')
        ax.set_xlim(0,max(pos[0]))
        ax.set_xticks([0,max(pos[0])])
        ax.set_ylabel('Y \u03BCm')
        ax.set_ylim(0,max(pos[1]))
        ax.set_yticks([0,max(pos[1])])
        ax.set_zlabel('Z \u03BCm')
        ax.set_zlim(0,max(pos[2]))
        ax.set_zticks([0,max(pos[2])])
        """Ruffling graph"""
        """ax.view_init(400,225,roll=None)"""
        """Phagocytosis graph"""
        ax.view_init(400,255,roll=None)
        fig.set_size_inches(5.5, 5.5)
        plt.savefig(frame,dpi=350)
    else:
        return surface.pathlength

    if track == True:
        x_coord_t, y_coord_t, z_coord_t = split(subtract(to_surface.vertices, centroid), 3, 1)
        """Defining coordinates for surface t+1"""
        x_coord_t = x_coord_t.flatten()
        y_coord_t = y_coord_t.flatten()
        z_coord_t = z_coord_t.flatten()
        """Converting the cartesian coordinates into spherical coordinates for surface t+1"""
        z_squared_t = z_coord_t ** 2
        y_squared_t = y_coord_t ** 2
        x_squared_t = x_coord_t ** 2

        distance_t = sqrt(z_squared_t + y_squared_t + x_squared_t)
        sphere_t = distance_t  < radius
        con_t = vertex_convexity(to_surface.vertices, to_surface.triangles, smoothing_iterations)

        ind_t = (con_t > thresh)

        car_t = zeros(shape(to_surface.vertices))

        car_t[:,0]= to_surface.vertices[:,0]*ind_t*sphere_t
        car_t[:,1]= to_surface.vertices[:,1]*ind_t*sphere_t
        car_t[:,2]= to_surface.vertices[:,2]*ind_t*sphere_t
        query_distance,_=query_tree(car, car_t, knn=knn)

        """The Average distance of the nearest neighbors"""
        surface.q_dist=query_distance
        return surface.q_dist
    else:
        return 

def ImgReconstruct(Points, x_coord, y_coord, z_coord, SearchLim, radius, size):
    """This script will reconstruct an image form the location of vertices in your rendered surface"""

    """Logic statements for specific objects we care about"""
    XYZ_SearchR = SearchLim
    XYZ_deletes = where(XYZ_SearchR==0)

    """Solving for unique X,Y,Z coordinates in the search"""
    xx=x_coord[:]*Points
    yy=y_coord[:]*Points
    zz=z_coord[:]*Points

    xyz_raw = column_stack((xx,yy,zz))

    """Reconstruct the image from points we define"""
    xyz=unique(delete(xyz_raw,XYZ_deletes,axis=0),axis=0)

    """Defining the pixel size from human defined parameter"""
    width = size[0]

    """Defining steps that will are approximately one pixel in length"""
    steps = int64(round_(abs((2*radius)/(width))))

    """Indexing the vertices that fall in one pixel of eachother along each axis""" 
    """nearest neighbors approach"""    
    xbins = digitize(xyz[:,0],linspace(-1*(radius),radius,steps))
    ybins = digitize(xyz[:,1],linspace(-1*(radius),radius,steps))
    zbins = digitize(xyz[:,2],linspace(-1*(radius),radius,steps))

    """Making an artificial binary mask of binned vertices into 'pixels' from location of vertices"""
    
    ArtImgxyz= zeros([steps,steps,steps])
    ArtImgxyz[xbins,ybins,zbins]= 1

    ArtImg = binary_erosion(((gaussian_filter(ArtImgxyz,.2))>0),border_value=1,iterations=1)
    ArtImg = ArtImg.astype('int8')
    return ArtImg

def measure_intensity(session, surface, to_map, radius, xnorm, ynorm, znorm, blob, output):
    """Measure the local intensity within radius r of the surface."""
    image_info = get_image(surface, to_map)
    masked_image = mask_image(*image_info)
    image_coords, *flattened_indices = get_coords(masked_image)
    _, index = query_tree(surface.vertices, image_coords.T, radius)
    face_intensity = local_intensity(*flattened_indices, index)
    surface.intensity = face_intensity

    """
    Amended section is designed to report and create palettes for 
    hemispheres on target of intensity values
    Author: YML
    Date: 20230907
    """
    if xnorm and ynorm and znorm is not None:
        if blob == 1:
            dust = largest_blobs_triangle_mask(surface.vertices, surface.triangles, surface.triangle_mask, blob_count=1, rank_metric = 'volume rank')

        elif blob == 2:
            dust_1st = largest_blobs_triangle_mask(surface.vertices, surface.triangles, surface.triangle_mask, blob_count=1, rank_metric = "volume rank")+0
            dust_2nd = largest_blobs_triangle_mask(surface.vertices, surface.triangles, surface.triangle_mask, blob_count=2, rank_metric = "volume rank")+0
            dust= dust_2nd - dust_1st

        elif blob == 3:    
            dust_1st = largest_blobs_triangle_mask(surface.vertices, surface.triangles, surface.triangle_mask, blob_count=2, rank_metric = "volume rank")+0
            dust_2nd = largest_blobs_triangle_mask(surface.vertices, surface.triangles, surface.triangle_mask, blob_count=3, rank_metric = "volume rank")+0
            dust= dust_2nd - dust_1st

        elif blob == 4:    
            dust_1st = largest_blobs_triangle_mask(surface.vertices, surface.triangles, surface.triangle_mask, blob_count=3, rank_metric = "volume rank")+0
            dust_2nd = largest_blobs_triangle_mask(surface.vertices, surface.triangles, surface.triangle_mask, blob_count=4, rank_metric = "volume rank")+0
            dust= dust_2nd - dust_1st

        
        rave = column_stack([(surface.triangles[:,0]*dust),(surface.triangles[:,1]*dust),(surface.triangles[:,2]*dust)]).flatten()
        
        vert_mask = zeros(shape(surface.vertices[:,0]))

        vert_mask[rave] = 1
        vert_mask3d=column_stack([vert_mask,vert_mask, vert_mask])

        t_ver = (surface.vertices * vert_mask3d)
        t_ver[t_ver==0] = nan 
        centroid = nanmean( t_ver , axis=0)

        ClipPlane= (xnorm* (centroid[0] - surface.vertices[:,0])) + (ynorm*(centroid[1] - surface.vertices[:,1])) + (znorm* (centroid[2] - surface.vertices[:,2])) 
        Top_hemisphere = (ClipPlane>0) +0
        Bottom_hemisphere = (ClipPlane<0) +0

        surface.intensity = face_intensity * vert_mask
        surface.ClipTop = face_intensity * Top_hemisphere * vert_mask
        surface.ClipBot = face_intensity * Bottom_hemisphere * vert_mask
        


        """Text file output"""
        path = exists(output)
        frame = surface.id[1]
        topsum = sum(surface.ClipTop)
        botsum = sum(surface.ClipBot)
        path = exists(output)

        if path == True:
            cd(session,str(output))
            with open('intensity.csv', 'ab') as f:
                savetxt(f, column_stack([frame, topsum, botsum]),
                        header=f"Frame Clip_Top Clip_Bot", comments='')
        else:
            return
        return surface.intensity, surface.ClipTop, surface.ClipBot
    else:
        return surface.intensity
    
def find_voids(session, surface, bg, sd, stg, dups, per , drop ):

    mask_vol = surface.volume.full_matrix().copy()
    mean = nanmean(mask_vol,where=(mask_vol> bg))
    dev = nanstd(mask_vol,where=(mask_vol> bg))

    masked_image= (mask_vol >= (mean-(dev*sd)))
    masking= (masked_image<1)

    blank=zeros(shape(masking))

    for n in range( shape(masked_image)[0] ):
        edge = canny(masked_image[n,:,:],sigma=1)
        blank[n,:,:] = edge
    


    d=binary_dilation(blank,iterations=1)

    dd=binary_dilation(blank,iterations=2)

    g= 1*dd-1*d

    p=binary_dilation(g,iterations=3)

    shave=masked_image*(p+d)

    y_l=gaussian_laplace((mask_vol*((shave+0) - (masking+0))),sigma=.3,truncate=3)
    y=gaussian_laplace((mask_vol),sigma=.3,truncate=8)

    stlg = 0.8125 * stg

    shadow=y
    shadow_thresh=shadow>=(stg*numpy.max(shadow))

    shadow_l=y_l
    shadow_thresh_l=shadow_l>=(stlg*numpy.max(shadow_l))

    img_shave = y*(shadow_thresh+shadow_thresh_l)

    heightMax = numpy.max(where(shave<=0.82),axis=1)[0]
    heightLimit= heightMax * 0.95

    if dups is not None:
        if dups == 1:
            img_i = img_shave >= (per*numpy.max(img_shave))
            dusted_i  =zeros(shape(img_shave))
            img_int1=img_i.astype("uint8")
            for n in range(shape(masked_image)[0]):
                void = label(img_int1[n,:,:],background=0)
                RegionID, RegionSize= unique(void,return_counts=True)
                exclusion=asarray(RegionSize>58).nonzero()    
                dusted_values=RegionID[exclusion]
                count=numpy.sum(exclusion)
                if count <= 10 or n>heightLimit:
                    dusted_i[n,:,:] = ones(shape(void))
                else:
                    dusted_i[n,:,:] = void==dusted_values.any()
            voids_found=((img_i-dusted_i))>0      
            mask=ArrayGridData((voids_found))
            volume_from_grid_data(mask,session)

        elif dups ==2:
            img_i = img_shave >= (per*numpy.max(img_shave))
            dusted_i  =zeros(shape(img_shave))
            img_int1=img_i.astype("uint8")
            for n in range(shape(masked_image)[0]):
                void = label(img_int1[n,:,:],background=0)
                RegionID, RegionSize= unique(void,return_counts=True)
                exclusion=asarray(RegionSize>58).nonzero()    
                dusted_values=RegionID[exclusion]
                count=numpy.sum(exclusion)
                if count <= 10 or n>heightLimit:
                    dusted_i[n,:,:] = ones(shape(void))
                else:
                    dusted_i[n,:,:] = void==dusted_values.any()
    
            img_ii = img_shave >= (((per-drop))*numpy.max(img_shave))
            dusted_ii  =zeros(shape(img_shave))
            
            img_int2=img_ii.astype("uint8")
            for n in range(shape(masked_image)[0]):
                void = label(img_int2[n,:,:],background=0)
                RegionID, RegionSize= unique(void,return_counts=True)
                exclusion=asarray(RegionSize>58).nonzero()    
                dusted_values=RegionID[exclusion]
                count=numpy.sum(exclusion)
                if count <= 10 or n>heightLimit:
                    dusted_ii[n,:,:] = ones(shape(void))
                else:
                    dusted_ii[n,:,:] = void==dusted_values.any()

            voids_found=((img_i-dusted_i) + (img_ii-dusted_ii))>0      
            mask=ArrayGridData((voids_found))
            volume_from_grid_data(mask,session)

        elif dups ==3:
            img_i = img_shave >= (per*numpy.max(img_shave))
            dusted_i  =zeros(shape(img_shave))
            img_int1=img_i.astype("uint8")
            for n in range(shape(masked_image)[0]):
                void = label(img_int1[n,:,:],background=0)
                RegionID, RegionSize= unique(void,return_counts=True)
                exclusion=asarray(RegionSize>58).nonzero()    
                dusted_values=RegionID[exclusion]
                count=numpy.sum(exclusion)
                if count <= 10 or n>heightLimit:
                    dusted_i[n,:,:] = ones(shape(void))
                else:
                    dusted_i[n,:,:] = void==dusted_values.any()

            img_ii = img_shave >= ((per-drop)*numpy.max(img_shave))
            dusted_ii  =zeros(shape(img_shave))
            
            img_int2=img_ii.astype("uint8")
            for n in range(shape(masked_image)[0]):
                void = label(img_int2[n,:,:],background=0)
                RegionID, RegionSize= unique(void,return_counts=True)
                exclusion=asarray(RegionSize>58).nonzero()    
                dusted_values=RegionID[exclusion]
                count=numpy.sum(exclusion)
                if count <= 10 or n>heightLimit:
                    dusted_ii[n,:,:] = ones(shape(void))
                else:
                    dusted_ii[n,:,:] = void==dusted_values.any()
            
            img_iii = img_shave >= ((per-(2*drop))*numpy.max(img_shave))
            dusted_iii  =zeros(shape(img_shave))
            img_int3=img_iii.astype("uint8")
            for n in range(shape(masked_image)[0]):
                void = label(img_int3[n,:,:],background=0)
                RegionID, RegionSize= unique(void,return_counts=True)
                exclusion=asarray(RegionSize>58).nonzero()    
                dusted_values=RegionID[exclusion]
                count=numpy.sum(exclusion)
                if count <= 10 or n>heightLimit:
                    dusted_iii[n,:,:] = ones(shape(void))
                else:
                    dusted_iii[n,:,:] = void==dusted_values.any()
            
            voids_found=((img_i-dusted_i) + (img_ii-dusted_ii) + (img_iii-dusted_iii))>0      
            mask=ArrayGridData((voids_found))
            volume_from_grid_data(mask,session)

        elif dups ==4:
            img_i = img_shave >= (per*numpy.max(img_shave))
            dusted_i  =zeros(shape(img_shave))
            img_int1=img_i.astype("uint8")
            for n in range(shape(masked_image)[0]):
                void = label(img_int1[n,:,:],background=0)
                RegionID, RegionSize= unique(void,return_counts=True)
                exclusion=asarray(RegionSize>58).nonzero()    
                dusted_values=RegionID[exclusion]
                count=numpy.sum(exclusion)
                if count <= 10 or n>heightLimit:
                    dusted_i[n,:,:] = ones(shape(void))
                else:
                    dusted_i[n,:,:] = void==dusted_values.any()

            img_ii = img_shave >= ((per-drop)*numpy.max(img_shave))
            dusted_ii  =zeros(shape(img_shave))
            
            img_int2=img_ii.astype("uint8")
            for n in range(shape(masked_image)[0]):
                void = label(img_int2[n,:,:],background=0)
                RegionID, RegionSize= unique(void,return_counts=True)
                exclusion=asarray(RegionSize>58).nonzero()    
                dusted_values=RegionID[exclusion]
                count=numpy.sum(exclusion)
                if count <= 10 or n>heightLimit:
                    dusted_ii[n,:,:] = ones(shape(void))
                else:
                    dusted_ii[n,:,:] = void==dusted_values.any()
                
            img_iii = img_shave >= ((per-(2*drop))*numpy.max(img_shave))
            dusted_iii  =zeros(shape(img_shave))
            img_int3=img_iii.astype("uint8")
            for n in range(shape(masked_image)[0]):
                void = label(img_int3[n,:,:],background=0)
                RegionID, RegionSize= unique(void,return_counts=True)
                exclusion=asarray(RegionSize>58).nonzero()    
                dusted_values=RegionID[exclusion]
                count=numpy.sum(exclusion)
                if count <= 10 or n>heightLimit:
                    dusted_iii[n,:,:] = ones(shape(void))
                else:
                    dusted_iii[n,:,:] = void==dusted_values.any()
            
            img_iiii = img_shave >= ((per-(3*drop))*numpy.max(img_shave))
            dusted_iiii  =zeros(shape(img_shave))
            img_int4=img_iiii.astype("uint8")
            for n in range(shape(masked_image)[0]):
                void = label(img_int4[n,:,:],background=0)
                RegionID, RegionSize= unique(void,return_counts=True)
                exclusion=asarray(RegionSize>58).nonzero()    
                dusted_values=RegionID[exclusion]
                count=numpy.sum(exclusion)
                if count <= 10 or n>heightLimit:
                    dusted_iiii[n,:,:] = ones(shape(void))
                else:
                    dusted_iiii[n,:,:] = void==dusted_values.any()    
            
            voids_found=((img_i-dusted_i) + (img_ii-dusted_ii) + (img_iii-dusted_iii) + (img_iiii-dusted_iiii))>0      
            mask=ArrayGridData((voids_found))
            volume_from_grid_data(mask,session)
    else:
        img_i = img_shave >= (per*numpy.max(img_shave))
        img_ii = img_shave >= ((per-drop)*numpy.max(img_shave))
        img_iii = img_shave >= ((per-(2*drop))*numpy.max(img_shave))
        img_iiii = img_shave >= ((per-(3*drop))*numpy.max(img_shave))

        dusted_i  =zeros(shape(img_shave))
        dusted_ii  =zeros(shape(img_shave))
        dusted_iii  =zeros(shape(img_shave))
        dusted_iiii  =zeros(shape(img_shave))
        
        img_int1=img_i.astype("uint8")
        for n in range(shape(masked_image)[0]):
            void = label(img_int1[n,:,:],background=0)
            RegionID, RegionSize= unique(void,return_counts=True)
            exclusion=asarray(RegionSize>58).nonzero()    
            dusted_values=RegionID[exclusion]
            count=numpy.sum(exclusion)
            if count <= 10 or n>heightLimit:
                dusted_i[n,:,:] = ones(shape(void))
            else:
                dusted_i[n,:,:] = void==dusted_values.any()
        
        img_int2=img_ii.astype("uint8")
        for n in range(shape(masked_image)[0]):
            void = label(img_int2[n,:,:],background=0)
            RegionID, RegionSize= unique(void,return_counts=True)
            exclusion=asarray(RegionSize>58).nonzero()    
            dusted_values=RegionID[exclusion]
            count=numpy.sum(exclusion)
            if count <= 10 or n>heightLimit:
                dusted_ii[n,:,:] = ones(shape(void))
            else:
                dusted_ii[n,:,:] = void==dusted_values.any()
        
        img_int3=img_iii.astype("uint8")
        for n in range(shape(masked_image)[0]):
            void = label(img_int3[n,:,:],background=0)
            RegionID, RegionSize= unique(void,return_counts=True)
            exclusion=asarray(RegionSize>58).nonzero()    
            dusted_values=RegionID[exclusion]
            count=numpy.sum(exclusion)
            if count <= 10 or n>heightLimit:
                dusted_iii[n,:,:] = ones(shape(void))
            else:
                dusted_iii[n,:,:] = void==dusted_values.any()

        img_int4=img_iiii.astype("uint8")
        for n in range(shape(masked_image)[0]):
            void = label(img_int4[n,:,:],background=0)
            RegionID, RegionSize= unique(void,return_counts=True)
            exclusion=asarray(RegionSize>58).nonzero()    
            dusted_values=RegionID[exclusion]
            count=numpy.sum(exclusion)
            if count <= 10 or n>heightLimit:
                dusted_iiii[n,:,:] = ones(shape(void))
            else:
                dusted_iiii[n,:,:] = void==dusted_values.any()

        voids_found=((img_i-dusted_i)+(img_ii-dusted_ii)+(img_iii-dusted_iii)+(img_iiii-dusted_iiii))>0
        mask=ArrayGridData((voids_found))
        volume_from_grid_data(mask,session)

def void_size(session, surface, track, tp, output):
    """import tract export size yml"""
    vol = surface.volume.full_matrix().copy()
    xyz= track[tp].coord
    f = xyz / .1028
    cen=(round(f)).astype('int')
    mask = vol + 0
    pixcc=label(mask)
    id = pixcc[ cen[2], cen[1], cen[0] ]  
    volume =count_nonzero(pixcc==id)* (.1028*.1028*.1028)

    cd(session, str(output))
    with open('volume.csv', 'ab') as f:
        savetxt(f, column_stack([volume,tp]),
            header=f"volume time_point", comments='')
    
def Void_Motion(session,track,t,output):
    l=numpy.zeros(numpy.shape(track.coords))[:,0]
    
    for n in range((numpy.shape(track.coords)[0])-1):
        l[n]=sqrt(sum((track.coords[n+1]-track.coords[n])**2))
    
    distance=sum(l)
    RMS= distance/sqrt(shape((track.coords)[0]))
    
    cd(session, str(output))
    with open('TrackMotion.csv', 'ab') as f:
        savetxt(f, column_stack([t,distance,RMS]),
            header=f"Track Distance Velocity/Frame", comments='')
    


def measure_composite(surface, green_map, magenta_map, radius):
    """Measure the local intensity for 2 channels within radius r of the surface."""
    green_coords, *green_indices = get_image_coords(surface, green_map)
    _, green_index = query_tree(surface.vertices, green_coords.T, radius)
    green_intensity = local_intensity(*green_indices, green_index)
    magenta_coords, *magenta_indices = get_image_coords(surface, magenta_map)
    _, magenta_index = query_tree(surface.vertices, magenta_coords.T, radius)
    magenta_intensity = local_intensity(*magenta_indices, magenta_index)
    surface.ch1 = green_intensity
    surface.ch2 = magenta_intensity


def get_image(surface, to_map):
    """Get the isosurface volume mask and secondary channel."""
    mask_vol = surface.volume.full_matrix().copy()
    image_3d = to_map.volume.full_matrix().copy()
    level = surface.volume.maximum_surface_level
    return mask_vol, level, image_3d


def get_coords(image_3d):
    """Get the coords for local intensity"""
    # ChimeraX uses XYZ for image, but numpy uses ZYX, swap dims
    image_3d = swapaxes(image_3d, 0, 2)
    image_coords = array(image_3d.nonzero())
    flattened_image = image_3d.flatten()
    pixel_indices = ravel_multi_index(image_coords, image_3d.shape)
    return image_coords, flattened_image, pixel_indices


def mask_image(mask, level, image_3d):
    """Mask the secondary channel based on the isosurface. Uses a 3D ball to dilate and erode with radius 2, then xor to make membrane mask."""
    mask = mask >= level
    struct_el = iterate_structure(generate_binary_structure(3, 1), 2)
    mask_d = binary_dilation(mask, structure=struct_el)
    mask_e = binary_erosion(mask, structure=struct_el, iterations=2)
    masked = mask_d ^ mask_e
    image_3d *= masked
    return image_3d


def query_tree(init_verts, to_map, radius=inf, knn=200):
    """Create a KDtree from a set of points and query for nearest neighbors.
    index: index of nearest neighbors within radius
    distance: Mean distance of nearest neighbors"""
    tree = KDTree(to_map)
    dist, index = tree.query(init_verts, k=range(
        1, knn), distance_upper_bound=radius, workers=-1)
    dist[dist == inf] = None
    distance = nanmean(dist, axis=1)
    index = array([_index(ind, tree.n) for ind in index], dtype=object)
    return distance, index


def _index(index, tree_max):
    """Tree query pads with tree_max if there are no neighbors."""
    index = index[index < tree_max]
    return index


def local_intensity(flat_img, pixels, index):
    """Measure local mean intensity normalized to mean of all."""
    face_int = array([nanmean(flat_img[pixels[ind]]) for ind in index])
    return face_int/face_int.mean()


def get_image_coords(surface, image):
    """Get the image coordinates for use in KDTree."""
    image_info = get_image(surface, image)
    masked_image = mask_image(*image_info)
    image_coords, *flattened_indices = get_coords(masked_image)
    return image_coords, *flattened_indices


def recolor_surface(session, surface, metric, palette, color_range, key):
    """Colors surface based on previously measured intensity or distance"""
    if metric == 'distance' and hasattr(surface, 'distance'):
        measurement = surface.distance
        palette_string = 'brbg'
        max_range = 15
    elif metric == 'intensity' and hasattr(surface, 'intensity'):
        measurement = surface.intensity
        palette_string = 'purples'
        max_range = 5
    elif metric == 'top' and hasattr(surface, 'ClipTop'):
        measurement = surface.ClipTop
        palette_string = 'purples'
        max_range = 5
    elif metric == 'bottom' and hasattr(surface, 'ClipBot'):
        measurement = surface.ClipBot
        palette_string = 'purples'
        max_range = 5
    elif metric == 'R' and hasattr(surface, 'radialDistance'):
        measurement = surface.radialDistance
        palette_string = 'purples'
        max_range = 100
    elif metric == 'Rphi' and hasattr(surface, 'radialDistanceAbovePhi'):
        measurement = surface.radialDistanceAbovePhi
        palette_string = 'purples'
        max_range = 10
    elif metric == 'rpg' and hasattr(surface, 'radialDistanceAbovePhiLimitxy'):
        measurement = surface.radialDistanceAbovePhiLimitxy
        palette_string = 'purples'
        max_range = 10
    elif metric == 'rpd' and hasattr(surface, 'radialDistanceAbovePhiNoNans'):
        measurement = surface.radialDistanceAbovePhiNoNans
        palette_string = 'purples'
        max_range = 10
    elif metric == 'theta' and hasattr(surface, 'theta'):
        measurement = surface.theta
        palette_string = 'brbg'
        if color_range is None:
            color_range = -pi,pi
    elif metric == 'phi' and hasattr(surface, 'phi'):
        measurement = surface.phi
        palette_string = 'brbg'
        max_range = pi
    elif metric == 'area' and hasattr(surface, 'areasearch'):
        measurement = surface.areasearch
        palette_string = 'purples'
        max_range = 1
    elif metric == 'edges' and hasattr(surface, 'edges'):
        measurement = surface.edges * 1
        palette_string = 'purples'
        max_range = 10
    elif metric == 'qd' and hasattr(surface, 'q_dist'):
        measurement = surface.q_dist.astype('int64')
        palette_string = 'brbg'
        max_range = 10
    else:
        return

    # If all the measurements are np.nan set them to zero.
    if isnan(measurement).all():
        measurement[:] = 0

    if palette is None:
        palette = colors.BuiltinColormaps[palette_string]

    if color_range is not None and color_range != 'full':
        rmin, rmax = color_range
    elif color_range == 'full':
        rmin, rmax = nanmin(measurement), nanmax(measurement)
    else:
        rmin, rmax = (0, max_range)

    cmap = palette.rescale_range(rmin, rmax)
    surface.vertex_colors = cmap.interpolated_rgba8(measurement)

    if key:
        show_key(session, cmap)


def composite_color(session, surface, palette, green_range, magenta_range, palette_range):
    """Colors surface based on previously measured intensity or distance"""
    if hasattr(surface, 'ch1') and hasattr(surface, 'ch2'):
        if palette == 'magenta_green':
            green_channel = surface.ch2
            magenta_channel = surface.ch1
        else:
            green_channel = surface.ch1
            magenta_channel = surface.ch2
    else:
        return

    # If all the measurements are np.nan set them to zero.
    if isnan(green_channel).all():
        green_channel[:] = 0
    if isnan(magenta_channel).all():
        magenta_channel[:] = 0

    green_range = scale_range(green_range, green_channel)
    magenta_range = scale_range(magenta_range, magenta_channel)

    # Define the color palettes.
    # gvals = ['#003c00', '#00b400']
    # mvals = ['#3c003c', '#b400b4']
    gmap = make_palette(green_channel, green_range, palette_range)
    mmap = make_palette(magenta_channel, magenta_range, palette_range,'magenta')

    # Build the composite vertex colors
    composite_map = array(gmap)
    composite_map[:, 1] = gmap[:, 1]
    composite_map[:, 0] = mmap[:, 0]
    composite_map[:, 2] = mmap[:, 2]

    surface.vertex_colors = composite_map

def make_palette(channel_data, color_range, palette_range, palette='green'):
    """Helper function to make the new color palette"""
    low, high = palette_range
    if palette == 'green':
        vals = [f'#00{low:02x}00', f'#00{high:02x}00']
    else:
        vals = [f'#{low:02x}00{low:02x}', f'#{high:02x}00{high:02x}']

    vals = [colors.Color(v) for v in vals]
    color = colors.Colormap(None, vals)
    palette = color.rescale_range(color_range[0],color_range[1])
    colormap = palette.interpolated_rgba8(channel_data)

    return colormap

def scale_range(color_range=(0,30), channel=None):
    """Helper function to set the color range"""
    if color_range == 'full':
        color_range = nanmin(channel), nanmax(channel)

    return color_range


measure_distance_desc = CmdDesc(
    required=[('surface', SurfacesArg)],
    keyword=[('to_surface', SurfacesArg),
             ('knn', Bounded(IntArg)),
             ('palette', ColormapArg),
             ('color_range', ColormapRangeArg),
             ('key', BoolArg)],
    required_arguments=['to_surface'],
    synopsis='Measure local distance between two surfaces')


measure_intensity_desc = CmdDesc(
    required=[('surface', SurfacesArg)],
    keyword=[('to_map', SurfacesArg),
             ('radius', Bounded(IntArg, 1, 30)),
             ('xnorm',FloatArg),
             ('ynorm',FloatArg),
             ('znorm',FloatArg),
             ('output',StringArg),
             ('blob', Bounded(IntArg, 1 , 4)),
             ('palette', ColormapArg),
             ('color_range', ColormapRangeArg),
             ('key', BoolArg)],
    required_arguments=['to_map'],
    synopsis='Measure local intensity relative to surface')


measure_composite_desc = CmdDesc(
    required=[('surface', SurfacesArg)],
    keyword=[('green_map', SurfacesArg),
             ('magenta_map', SurfacesArg),
             ('radius', Bounded(IntArg, 1, 30)),
             ('green_range', ColormapRangeArg),
             ('magenta_range', ColormapRangeArg)],
    required_arguments=['green_map', 'magenta_map'],
    synopsis='Measure local intensities of two channels relative to surface')


recolor_surfaces_desc = CmdDesc(
    required=[('surface', SurfacesArg)],
    keyword=[('metric', EnumOf(['intensity', 'distance','R', 'theta', 'phi', 'Rphi', 'rpg','rpd', 'area', 'edges', 'qd', 'bottom','top'])),
             ('palette', ColormapArg),
             ('color_range', ColormapRangeArg),
             ('key', BoolArg)],
    required_arguments=[],
    synopsis='Recolor surface based on previous measurement')


recolor_composites_desc = CmdDesc(
    required=[('surface', SurfacesArg)],
    keyword=[('palette', EnumOf(['green_magenta', 'magenta_green'])),
             ('green_range', ColormapRangeArg),
             ('magenta_range', ColormapRangeArg),
             ('palette_range', Int2Arg)],
    required_arguments=[],
    synopsis='Recolor surface based on previous measurements as a composite')

measure_topology_desc = CmdDesc(
    required=[('surface', SurfacesArg)],
    keyword=[('to_cell', SurfacesArg),
             ('metric', EnumOf(['R', 'theta', 'phi', 'Rphi', 'rpg','rpd', 'area'])),
             ('palette', ColormapArg),
             ('radius', Bounded(IntArg)),
             ('target', EnumOf(['sRBC', 'mRBC'])),
             ('color_range', ColormapRangeArg),
             ('phi_lim', Bounded(IntArg)),
             ('output', StringArg),
             ('key', BoolArg)],
    required_arguments=['to_cell'],
    synopsis='This measure function will output calculated axial surface roughness values (S_q)'
        'based on inputs surface-Macrophage, tocell- target, radius- search radius (um), targetr- target radius (um)')

measure_ridges_desc = CmdDesc(
    required=[('surface',SurfaceArg)],
    keyword=[('to_surface', SurfaceArg),
             ('to_cell', SurfaceArg),
             ('smoothing_iterations', Bounded(IntArg)),
             ('thresh', Bounded(FloatArg)),
             ('palette', ColormapArg),
             ('radius', Bounded(FloatArg)),
             ('knn', Bounded(IntArg)),
             ('color_range', ColormapRangeArg),
             ('clip', Bounded(FloatArg)),
             ('output', StringArg),
             ('track', BoolArg),
             ('exclusion', Bounded(FloatArg)),
             ('key', BoolArg)],
    required_arguments = ['to_surface','to_cell'],
    synopsis = 'Current implimentation focuses on identifying high curvature'
        'lamella edges for video recodings')

find_voids_desc = CmdDesc(
    required=[('surface',SurfaceArg)],
    keyword=[('bg', FloatArg),
             ('sd', Bounded(FloatArg,0,1)),
             ('stg', Bounded(FloatArg,0,1)),
             ('dups', Bounded(IntArg,1,4)),
             ('per', Bounded(FloatArg,0,1)),
             ('drop', Bounded(FloatArg,0,.99))],
    synopsis = 'This function is designed to find voids in a surface rendering.')

void_size_desc = CmdDesc(
    required=[('surface', SurfaceArg)],
    keyword=[('track',AtomsArg),
             ('tp',IntArg), 
             ('output',StringArg)],
    required_arguments = ['surface','track','tp','output'],
    synopsis='output volume')

Void_Motion_desc = CmdDesc(
    required=[('track', AtomsArg)],
    keyword=[('t',IntArg),
             ('output',StringArg)],
    required_arguments = ['track','t','output'],
    synopsis='output volume')
