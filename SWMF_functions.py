import numpy as np
import os
import datetime
import sys
from spacepy import pybats as pb
import spacepy.pybats.bats as bt
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from spacepy import coordinates as coord
from spacepy.time import Ticktock
from scipy.interpolate import griddata

############################################################################

def get_datetime_magfile(f):
    
    """return datetime object from mag filename f"""
    
    f = f.split("/")[-1]    # look at mag_grid*.out part
    b = f[10:]
    td = datetime.datetime.strptime(b, "%Y%m%d-%H%M%S.out")

    return td

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

def read_in_raw_SWMF_magfiles(infold, inputcols, outfold='', outprefix = ''):

    """Read in all SWMF mag*.out files as dictionary, and optionally save outputs as individual numpy objects for later use

    Parameters
    -----------
    infold = location of folder with mag files (usually 'GM/IO2/')
    inputcols = list of wanted variables, e.g: ['bx', 'by', 'bz']
        possible variables: bx, by, bz, mx, my, mz, fx, fy, fz, hx, hy, hz, px, py, pz
    outfold = location of folder to save outputs. Leave as '' (the default) to not save 
    outputprefix = optional prefix when saving outputs, '' by default

    Returns
    -----------
    if outfold == '' (as default):
        outputdata = dictionary of 'Lonlat', 'Timedate', and each of the columns asked for

    otherwise:
        returns outputdata and saves to whatever outfold was specified as"""
        
    # get the column index for the supplied variables
    cols = np.array(['Bx', 'By', 'Bz', 'Mx', 'My', 'Mz', 'Fx', 'Fy', 'Fz', 'Hx', 'Hy', 'Hz', 'Px', 'Py', 'Pz'])
    colnums = np.array([np.where(cols == i)[0][0]+2 for i in inputcols])

    filenames = sorted(os.listdir(infold))
    files_mag = [x for x in filenames if 'mag_grid' in x]

    timedate = np.array([get_datetime_magfile(x) for x in files_mag])   # get timedates

    lenfiles = len(files_mag) * 1.0
    lonlat = np.loadtxt(infold + files_mag[0], usecols = (0, 1), unpack = True, skiprows = 4).T

    template = np.ones((int(len(files_mag)), len(lonlat)))  # template output arrays
    outputdata = [template]*len(inputcols)

    print("Reading in files")
    for i1, v1 in enumerate(files_mag):
        data = np.loadtxt(infold + v1, skiprows=4, usecols=colnums, unpack=True)
        
        for i2, v2 in enumerate(data):
            outputdata[i2][i1] = v2
            
        print(i1)

    if outfold != '':
        print("Saving as numpy objects")
        for i, v in enumerate(outputdata):
            np.save(outfold + outprefix + inputcols[i], v)
            
        np.save(outfold + outprefix + 'lonlat', lonlat)
        np.save(outfold + outprefix + 'timedate', timedate)
        
    outdata = {}
    outdata['Timedate'] = timedate
    outdata['LonLat'] = lonlat
    for i, v in enumerate(inputcols):
        outdata[v] = outputdata[i]         
        
    return outdata    
        
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

def read_in_SWMF_magfiles(infolder):

    """Read in previously numpified mag outputs files as dictionary

    Parameters
    -----------
    infold = location of folder with mag files (usually GM/IO2/)
    inputcols = list of wanted variables, e.g: ['bx', 'by', 'bz']
        possible variables: bx, by, bz, mx, my, mz, fx, fy, fz, hx, hy, hz, px, py, pz
    outfold = location of folder to save outputs. Leave as '' (the default) to not save 
    outputprefix = optional prefix when saving outputs. '' by default

    Returns
    -----------
    if outfold == '' (as default):
        outputdata = dictionary of 'Lonlat', 'Timedate', and each of the magnetic data files

    otherwise:
        returns outputdata and saves to whatever outfold was specified as"""


    filenames = sorted(os.listdir(infolder))
    filenames = [x for x in filenames if '.npy' in x]

    outdata = {}

    timedatefile = [x for x in filenames if 'timedate.npy' in x][0]
    timedate = np.load(infolder + timedatefile, allow_pickle = True)
    outdata['Timedate'] = timedate
    
    lonlatfile = [x for x in filenames if 'lonlat.npy' in x][0]
    lonlat = np.load(infolder + timedatefile, allow_pickle = True)
    outdata['LonLat'] = lonlat
    
    cols = np.array(['Bx', 'By', 'Bz', 'Mx', 'My', 'Mz', 'Fx', 'Fy', 'Fz', 'Hx', 'Hy', 'Hz', 'Px', 'Py', 'Pz'])
    for i in cols:
        for j in filenames:
            if i in j:
                data = np.load(infolder + j)
                outdata[i] = np.copy(data)
    return outdata

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

def read_SWMF_Dst(folder):

    """Read in dst from multiple SWMF log* files
    
    Parameters
    -----------
    folder = location of folder which contain all the log* files    
    
    Returns
    -----------
    dst_timedate = array of timedates
    dst_vals = array of dst values"""
    
    filenames = [x for x in sorted(os.listdir(folder)) if "log_e" in x]
    outtime, outdst = [], []

    for i in filenames:
        a = pb.LogFile(folder + i)
        outtime.extend(a['time'])
        outdst.extend(a['dst'])

    return np.array(outtime), np.array(outdst)

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

def solar_wind_reader(filename, return_data = True, makeplot = False, skiprow = 9):

    """Quickly read and plot a SWMF formatted solar wind file to check for shenanigans

    Parameters
    -----------
    filename = the .dat SWMF solar wind input filename    
    return _data = if True, returns dictionary of solar wind values. True by default.
        outputs = Timedate, Bx, By, Bz, Vx, Vy, Vz, N, T, P 
    makeplot = if True, makes a plot of the SW conditions. False by default
    skiprow = number of rows to skip at top of file when reading in. 9 by default.
    
    Returns
    -----------
    outputdata = dictionary of data (if return_data == True)
    pretty plot of SW conditions (if makeplot == True)""" 
    
    data = np.loadtxt(filename, skiprows = skiprow)

    yy, mm, dd = data[:,0], data[:,1], data[:,2]
    HH, MM, SS = data[:,3], data[:,4], data[:,5]

    td = []
    for y, m, d, H, M, S in zip(yy, mm, dd, HH, MM, SS):
        qwe = datetime.datetime(int(y), int(m), int(d), int(H), int(M), int(S))
        td.append(qwe)
    td = np.array(td)

    bx, by, bz = data[:,7], data[:,8], data[:,9]
    vx, vy, vz = data[:,10], data[:,11], data[:,12]

    n, T = data[:,13], data[:,14]
    P = n * 1.6e-6 * vx * vx    # calculate pressure

    labels = ['Timedate', 'Bx', 'By', 'Bz', 'Vx', 'Vy', 'Vz', 'N', 'T', 'P']
    outputdata = {}
    for name, val in zip(labels, [td, bx, by, bz, vx, vy, vz, n, T, P]):
        outputdata[name] = val

    if makeplot == True:
        plt.clf()

        ax1 = plt.subplot(511)

        plt.plot(td, bx, label = 'bx')
        plt.plot(td, by, label = 'by')
        plt.plot(td, bz, label = 'bz')
        plt.legend()
        plt.ylabel("B (nT)", fontsize = 20)
        plt.grid(True)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

        ax2 = plt.subplot(512)
        plt.plot(td, vx, label = 'vx')
        plt.plot(td, vy, label = 'vy')
        plt.plot(td, vz, label = 'vz')
        plt.legend()
        plt.ylabel("V (km/s)", fontsize = 20)
        plt.grid(True)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

        ax3 = plt.subplot(513)
        plt.plot(td, n, label = 'N')
        plt.legend()
        plt.grid(True)
        plt.ylabel(r"N (/cm$^{3}$)", fontsize = 20)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

        ax4 = plt.subplot(514)
        plt.plot(td, T, label = 'T')
        plt.legend()
        plt.grid(True)
        plt.ylabel(r"T (K)", fontsize = 20)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

        ax5 = plt.subplot(515)
        plt.plot(td, P, label = 'P')
        plt.legend()
        plt.ylabel(r"P (nPa)", fontsize = 20)
        plt.grid(True)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.show()

    if return_data == True:
        return outputdata
        
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

def slice2d_plotter(filename):

    """Plot total J in magnetosphere for an SWMF 2d slice (filename)"""
    
    data2d = bt.Bats2d(filename)

    jx, jy, jz =  np.array(data2d['jx']), np.array(data2d['jy']), np.array(data2d['jz'])
    J = np.sqrt(jx**2 + jy**2 + jz**2)
    data2d['J'] = J

    plt.clf()
    ax1 = plt.subplot(111)
    ax1.set_aspect('equal', 'box')
    data2d.add_contour('x', 'z', 'J', target = ax1, dolog = True)

    plt.grid(True)

    plt.xlim([-20, 20])
    plt.ylim([-20, 20])

    data2d.add_b_magsphere(target = ax1)

    plt.show()

    return data2d

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

def get_magpause_distance(folder, makeplot = False, imagefolder = ''):

    """Calculate minimum magnetopause standoff distance from BATSRUS y=0 files  

    Parameters
    -----------
    folder = folder that contains y=0*.out files OR individual file
    makeplot = set to True to make plot of total J in magnetosphere for reference. False by default
    imagefolder = folder location to save images. Saves as 0000.png, 0001.png, etc.
    
    Returns
    -----------
    TD = array of datetimes
    MPAUSE = array of magnetopause distances
    fig = last figure (only if makeplot == True)"""
    
    TD, MPAUSE = [] ,[]
    if folder[-1] == '/':
        filenames = [folder + x for x in sorted(os.listdir(folder)) if 'y=0' in x]
    else:
        filenames = [folder]

    for iii, f in enumerate(filenames):

        # get timedate for the slice
        f2 = f.split("/")[-1][11:26]
        td = datetime.datetime.strptime(f2, "%Y%m%d-%H%M%S")
        TD.append(td)
        
        # get the XZ-coordinates of the last closed fieldlines
        B = bt.Bats2d(f)
        fline = B.find_earth_lastclosed()[3]
        flinex, flinez = np.array(fline['x']), np.array(fline['z'])

        # get angle for each point
        anglez = np.rad2deg(np.arctan(flinez/flinex))
        ind = (anglez > -25) * (anglez < 25)    # only want points mostly on sun-earth line

        # get point at max distance
        distfromorigin = np.sqrt(flinex**2 + flinez**2)
        mpause_dist = np.max(distfromorigin[ind])   
        mpause_ind = np.where(distfromorigin == mpause_dist)[0][0]
        MPAUSE.append(mpause_dist)
        
        # make plot of total J
        x, z = np.array(B['x']), np.array(B['z'])
        jx, jy, jz = np.array(B['jx']), np.array(B['jy']), np.array(B['jz'])
        J = np.sqrt(jx**2 + jy**2 + jz**2)
        B['J'] = J

        print(td, "   Re = %0.3f" % mpause_dist)
        
        if makeplot != False:
            fig = plt.figure(1, figsize = (8, 8))
            plt.clf()

            ax1 = plt.subplot(111)

            plt.title(str(td) + ", R = %0.2f" % mpause_dist, fontsize = 18)
            ax1.set_aspect('equal', 'box')
            B.add_contour('x', 'z', 'J', target = ax1, dolog=True)
            B.add_b_magsphere(target = ax1)
            plt.scatter(flinex[mpause_ind], flinez[mpause_ind], color = 'red', marker = '*', s = 200, zorder = 100)

            plt.grid(True)
            plt.xlim([-20, 20])
            plt.ylim([-20,20])

            plt.tight_layout()
            plt.savefig(imagefolder + "%04d.png" % iii)
            #plt.show()
    if makeplot == False:
        return np.array(TD), np.array(MPAUSE)
    else:
        return np.array(TD), np.array(MPAUSE), fig

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

def rbody_calculator(Rbody, td, glon, glat):

    """Calculate a GSM XYZ point on Rbody given a geo-lon and geo-lat
    Extrapolates to Rbody from the surface along assumed dipole lines

    **Not 100% confident in this calculation**
    
    Parameters
    -----------
    Rbody = Earth radii at which to calculate point
    td = datetime of interest
    glon, glat = longitude in degrees
    
    Returns
    -----------
    [X,Y,Z] = Cartesian GSM coordinates in Re"""
    
    # Convert GEO data to MAG
    cvals = coord.Coords([1, glat, glon], 'GEO', 'sph')
    cvals.ticks = Ticktock(td, 'UTC') # add ticks
    a = cvals.convert('MAG','sph')
    mlat, mlon = a.data[0][1], a.data[0][2]

    # get L-shell
    L = 1./np.cos(np.deg2rad(mlat))**2

    # construct dipole field lines
    theta = np.linspace(0, np.pi, 1000)

    r = L * np.sin(theta) **2
    x = r * np.cos(np.pi/2. - theta)
    y = r * np.sin(np.pi/2. - theta)

    # Get points of intersection with Rcurr boundary
    R = np.sqrt(x**2 + y**2)
    diff = np.abs(Rbody - R)
    ind = np.argpartition(diff, 2)[:2]

    intersecx = x[ind]
    intersecy = y[ind]
    qqq = [intersecy > 0]
    intersecx, intersecy = intersecx[qqq], intersecy[qqq]

    ANGLE = np.rad2deg(np.arcsin(intersecy/Rbody))

    # convert to GSE XYZ
    cval = coord.Coords([Rbody, ANGLE[0], mlon], 'MAG', 'sph')
    cval.ticks = Ticktock(td, 'UTC')
    c = cval.convert("GSM", 'car')

    return c.data[0]
    

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

def xyz_to_lonlat(x,y,z):
    """Convert cartesian points to latitude longitude"""

    XsqPlusYsq = x**2 + y**2
    r = np.sqrt(XsqPlusYsq + z**2)               # r
    #elev = np.arccos(z/np.sqrt(XsqPlusYsq))     # theta
    elev = np.arccos(z/r)
    az = np.arctan2(y,x)                           # phi
    return np.rad2deg(az), 90 - np.rad2deg(elev)   # want 

def cart2sphvec(x, y, z, az, el, degrees = True):

    """Convert cartesian vectors to spherical vectors"""
    
    if degrees == True:
        el = np.deg2rad(el)
        az = np.deg2rad(az)

    Vr = (np.cos(el) * np.cos(az) * x) + (np.cos(el) * np.sin(az) * y) + (np.sin(el) * z)
    Vaz = (-1 * np.sin(az) * x) + (np.cos(az) * y)
    Vel = (-1 * np.sin(el) * np.cos(az) * x) + (-1 * y * np.sin(el) * np.sin(az)) + (z * np.cos(el))

    return (Vaz, Vel, Vr)

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

def Jaz_plane_plotter(filename, imagefolder, slices = np.arange(-4, 4.01, .1)[::-1]):
    
    """Plot azimuthal J in different magnetospheric planes and save images

    Parameters
    -----------
    filename = SWMF 3d*.out file
    imagefolder = location to save images of slices
    slices = XY XZ planes at which to make images. -4 to 4 in steps of 0.1 by default"""
    
    data3d = bt.Bats2d(filename)

    x,y,z = np.array(data3d['x']), np.array(data3d['y']), np.array(data3d['z'])
    jx,jy,jz = np.array(data3d['jx']), np.array(data3d['jy']), np.array(data3d['jz'])
    jh = np.sqrt(jx**2 + jy**2)

    az, el = xyz_to_lonlat(x, y, z)
    Jaz, Jel, Jr = cart2sphvec(jx, jy, jz, az, el)

    indd = (x>-20)*(x<20)*(y>-20)*(y<20)*(z>-20)*(z<20) # reduce the data
    maxx, minn = np.max(Jaz), np.min(Jaz)

    fig = plt.figure(1, figsize = (20, 10))
    for i, v in enumerate(slices):
        print("X, Y = %0.2f Re" % v)
        wantedz = v

        # XY-plane # ------------------
        ind = (z < (wantedz+1)) * (z > (wantedz-1))
        x2, y2, z2 = x[ind], y[ind], z[ind]
        points, values = np.column_stack((x2, y2, z2)), Jaz[ind]

        newx, newy = np.arange(-20, 20, 0.2), np.arange(-20, 20, 0.2)
        uuu, vvv = np.meshgrid(newx, newy)
        uuu, vvv = uuu.flatten(), vvv.flatten()

        www = np.ones(len(uuu)) * wantedz
        wanted_points = np.array(list(zip(uuu, vvv, www)))
        interpped = griddata(points, values, wanted_points, method='nearest')

        # XZ-plane # ------------------
        ind = (y < (wantedz+1)) * (y > (wantedz-1))
        x3, y3, z3 = x[ind], y[ind], z[ind]
        points3, values3 = np.column_stack((x3, y3, z3)), Jaz[ind]

        newx3, newz3 = np.arange(-20, 20, 0.2), np.arange(-20, 20, 0.2)
        uuu3, vvv3 = np.meshgrid(newx3, newz3)
        uuu3, vvv3 = uuu3.flatten(), vvv3.flatten()

        www3 = np.ones(len(uuu3)) * wantedz
        wanted_points3 = np.array(list(zip(uuu3, www3, vvv3)))
        interpped3 = griddata(points3, values3, wanted_points3, method='nearest')

        # make plot
        maxx = np.max([np.max(Jaz), -1*np.min(Jaz)])
        levels = np.linspace(-1*maxx, maxx, 100)
        r1 = np.sqrt(1-wantedz**2)

        clf()

        ax1 = subplot(121)
        ax1.set_aspect('equal', 'box')
        title("Z = %.2f" % wantedz, fontsize = 24)
        tricontourf(uuu, vvv, interpped, levels = levels, cmap = 'RdBu_r')
        cb = colorbar()
        cb.set_label("$J_{Az}$", fontsize = 24)
        grid(True)

        xlim([-10, 10])
        ylim([-10, 10])

        xlabel("X ($R_E$)", fontsize = 20)
        ylabel("Y ($R_E$)", fontsize = 20)
        circle1 = plt.Circle((0, 0), r1, color='k')
        ax1.add_artist(circle1)
            
        ax2 = subplot(122)
        ax2.set_aspect('equal', 'box')
        title("Y = %.2f" % wantedz, fontsize = 24)

        tricontourf(uuu3, vvv3, interpped3, levels = levels, cmap = 'RdBu_r')
        cb = colorbar()
        cb.set_label("$J_{Az}$", fontsize = 24)
        grid(True)

        circle1 = plt.Circle((0, 0), r1, color='k')
        ax2.add_artist(circle1)

        xlim([-10, 10])
        ylim([-10, 10])

        xlabel("X ($R_E$)", fontsize = 20)
        ylabel("Z ($R_E$)", fontsize = 20)

        savefig(imagefolder + "%04d.png" % i)
        
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-





























































