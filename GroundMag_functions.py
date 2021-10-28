# functions to read in SuperMag and Intermagnet data from local storage
# also calculates E-fields using 1D plane-wave method and a reference 

import numpy as np
import math
import datetime
import os
import netCDF4 as nc
import spacepy.coordinates as coord
from spacepy.time import Ticktock
from gcvspline import SmoothedNSpline

###############################################################################
# point to where the SuperMag and INTERMAGNET data are
supermagdatafolder = '/media/blake/2TB_HDD/Datasets/SUPERMAG/DATA/'
INTERMAGdatafolder = '/media/blake/2TB_HDD/Datasets/INTERMAGNET_DATA/IAGA2002_DEF/'
DSTfolder = '/media/blake/2TB_HDD/Datasets/DST_SYMH/'
###############################################################################

def SuperMag_data_local(start, end, badfrac = 0.1, calc_efield = True):

    """Parameters
    -----------
    start, end = datetime objects to find data
    badfrac = fraction of nans that are acceptable. Time-series with fractions greater than this not returned
    calc_efield = If True, returns ex,ey,eh as calculated using the Quebec profile

    Returns
    -----------
    OUTPUT = dictionary of variables by sitecode"""

    start_year = start.year
    fn = supermagdatafolder + 'all_stations_none%04d.netcdf' % start_year
    ds = nc.Dataset(fn)
    lenn = ds['id'].shape[0]

    # Get sitenames
    id_raw = np.array(ds['id'][0]).astype(str)
    sitenames = np.array([x[0]+x[1]+x[2] for x in id_raw])

    # Read in timedate
    yr = np.array(ds['time_yr'][:])
    mo = np.array(ds['time_mo'][:])
    dy = np.array(ds['time_dy'][:])
    hr = np.array(ds['time_hr'][:])
    mt = np.array(ds['time_mt'][:])
    timedate = np.array([datetime.datetime(y,m,d,H,M,0) for y,m,d,H,M in zip(yr, mo, dy, hr, mt)])
    ind = (timedate >= start) * (timedate <= end)
    timedate = timedate[ind]

    # get lat,lon, mlat, mlon
    glat = np.array(ds['glat'][0])
    glon = np.array(ds['glon'][0])
    glon[glon >180] -= 360      # have 
    mlat = np.array(ds['mlat'][0])
    mlon = np.array(ds['mlon'][0])

    # magnetic local time
    mlt = np.array(ds['mlt'][ind]).T

    # read in bx, by (geographic vector)
    bx = np.array(ds['dbn_nez'][ind]).T
    by = np.array(ds['dbe_nez'][ind]).T
    bh = np.sqrt(bx**2 + by**2)
    bz = np.array(ds['dbz_nez'][ind]).T

    # Check for nans
    goodind, lenn = [], len(timedate)
    for i, v in enumerate(bh):
        if np.sum(np.isnan(v)) >= (badfrac * lenn):
            goodind.append(False)
        else:
            goodind.append(True)
            
    goodind = np.array(goodind)

    Qres, Qthick = model_profiles("Q")
    freq = np.fft.fftfreq(numofdays * 1440, d = 60.)
    freq[0] = 1e-100
    ZZ = Z_Tensor_1D(Qres, Qthick, freq)

    OUTPUT = {}
    for i, v in enumerate(sitenames[goodind]):
        gn, gt, mn, mt = glon[goodind][i], glat[goodind][i], mlon[goodind][i], mlat[goodind][i]
        td, mltt = timedate, mlt[goodind][i]
        x, y, z, h = bx[goodind][i], by[goodind][i], bz[goodind][i], bh[goodind][i]
        
        if calc_efield == False:
            OUTPUT[v] = {'glon':gn, 'glat':gt, 'mlon':mn, 'mlat':mt, 
                    'mlt':mltt, 'timedate':td, 'bx':x, 'by':y, 
                    'bz':z, 'bh':h}
        else:
            # need to interpolate bx and by, get rid of nans
            X = np.arange(len(x))
            ind = np.isfinite(x)
            newbx = np.interp(X, X[ind], x[ind])
            newby = np.interp(X, X[ind], y[ind])
            
            ex, ey = E_Field_1D(newbx, newby, Qres, Qthick, 60,  ZZ)
            eh = np.sqrt(ex**2 + ey**2)
            
            OUTPUT[v] = {'glon':gn, 'glat':gt, 'mlon':mn, 'mlat':mt, 
                    'mlt':mltt, 'timedate':td, 'bx':x, 'by':y, 
                    'bz':z, 'bh':h, 'ex':ex, 'ey':ey, 'eh':eh}

    return OUTPUT
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
def FetchSMIndices(user, start, numofdays, wanted = 'ALL'):
    """Retrieve SME, SML, SMU indices from SuperMag

    This function requires the supermag api functions to be loaded in memory to fetch the data:
    https://supermag.jhuapl.edu/mag/?fidelity=low&tab=api&start=2001-01-01T00%3A00%3A00.000Z&interval=23%3A59
    
    Parameters
    -----------
    user = username for downloading SuperMag Data
    start = start day (datetime obj)
    numofdays = number of days from start to download
    wanted = list of wanted attrs (e.g., ['SME', 'SML']
        downloads all by default

    Returns
    -----------
    output = dictionary of wanted values as arrays + 'td' array
    """
    #ZZZ
    status, vals = SuperMAGGetIndices(user, start, 86400*numofdays, 'all', FORMAT='list')

    if (wanted == 'ALL'):
        wanted = list(vals[0].keys())[1:]

    output = {x:[] for x in wanted}
    output['td'] = []

    for step in vals:
        output['td'].append(Float2Time(step['tval']))

        for j in wanted:
            output[j].append(step[j])

    ind = np.array(output['td']) >= start

    for i in output.keys():
        output[i] = np.array(output[i])[ind]

    return output
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

def FetchSMData(user, start, numofdays, savefolder, badfrac=0.1, nanflags=True):
    """Retrieve all available SuperMagnet data for a specified period
    If data has not already been downloaded, fetches data from Supermag
    
    This function requires the supermag api functions to be loaded in memory to fetch the data:
    https://supermag.jhuapl.edu/mag/?fidelity=low&tab=api&start=2001-01-01T00%3A00%3A00.000Z&interval=23%3A59
    
    Parameters
    -----------
    user = username for downloading SuperMag Data
    start = start day (datetime obj)
    numofdays = number of days from start to download
    savefolder = folder where downloaded data will be saved as json. This function
        looks here first for saved data before downloading.
    badfrac = tolerable fraction of data that is 99999.0. Sites with more bad data
        than this fraction will be ignored
    nanflags = will set 99999.0 values to nans if True (True by default)    

    Returns
    -----------
    Dictionary which has the following data as keys:
    {td, sitenames, glon, glat, mlon, mlat, mcolat, 
                BNm, BEm, BZm, BNg, BEg, BZg, MLT, DECL, SZA}
    """
    # Look at all saved .jsons
    filenames = [x for x in sorted(os.listdir(savefolder)) if '.json' in x]
    startstr = str(start)[:10]

    exists = False
    for filename in filenames:
        if startstr in filename:
            daystring = filename[:-5].split("_")[-1]

            if int(daystring) >= numofdays:
                print("Supermag data already exists locally")
                print(filename)
 
                exists = True
                break
            continue

    if exists == False:
        print("Supermag data not local, fetching:")

        STATUS, master, badindex = [], [], []

        #ZZZ
        status, stations = SuperMAGGetInventory(user, startstr, extent = 86400*numofdays)
        for iii in stations:
            print("Fetching: ", iii)
            #ZZZ
            status, A = SuperMAGGetData(user, startstr, extent=86400*numofdays, 
                                           flagstring='all', station = iii, FORMAT = 'list')
            quickvals = np.array([x['N']['nez'] for x in A])

            # get rid of data if too many bullshit values
            if np.sum(quickvals>999990.0) >= badfrac*len(quickvals):
                badindex.append(False)
                print(iii, "BAD")
            else:
                badindex.append(True)

            STATUS.append(status)
            master.append(A)

        badindex = np.array(badindex)
        master, stations = np.array(master)[badindex], np.array(stations)[badindex]

        # Make the Supermag data a dict for saving later
        output = {}
        for i in master:
            output[i[0]['iaga']] = list(i)

        filename = "SM_DATA_" + str(start)[:10] + '_%1d.json' % (numofdays)
        with open(savefolder + filename, mode='w') as f:
            #print(savefolder + filename)
            json.dump(output, f)
        f.close()

    # Now read in the data
    with open(savefolder + filename,) as r:
        #print(savefolder + filename)
        rr = json.load(r)

    sitenames = np.array(list(rr.keys()))

    timedate = np.array([Float2Time(x['tval']) for x in rr[sitenames[0]]])
    glon = np.array([rr[x][0]['glon'] for x in sitenames])
    glon[glon>180] -= 360 
    glat = np.array([rr[x][0]['glat'] for x in sitenames])
    mlon = np.array([rr[x][0]['mlon'] for x in sitenames])
    mlat = np.array([rr[x][0]['mlat'] for x in sitenames])
    mcolat = np.array([rr[x][0]['mcolat'] for x in sitenames])

    BNm = np.zeros((len(timedate), len(sitenames)))

    BEm, BZm, BNg, BEg, BZg = np.copy(BNm), np.copy(BNm), np.copy(BNm), np.copy(BNm), np.copy(BNm)
    MLT, DECL, SZA = np.copy(BNm), np.copy(BNm), np.copy(BNm)

    for j, v2 in enumerate(sitenames):
        for i, v1 in enumerate(rr[v2]):
            BNm[i][j] = v1['N']['nez']
            BEm[i][j] = v1['E']['nez']
            BZm[i][j] = v1['Z']['nez']
            
            BNg[i][j] = v1['N']['geo']
            BEg[i][j] = v1['E']['geo']
            BZg[i][j] = v1['Z']['geo']

            MLT[i][j] = v1['mlt']
            DECL[i][j] = v1['decl']
            SZA[i][j] = v1['sza']

    if (nanflags == True):
        BNm[BNm==999999.0] = np.nan
        BEm[BEm==999999.0] = np.nan
        BZm[BZm==999999.0] = np.nan

        BNg[BNg==999999.0] = np.nan
        BEg[BEg==999999.0] = np.nan
        BZg[BZg==999999.0] = np.nan

    # only use points after start of sim data
    i = (timedate>=start)
    output = {'td':timedate[i], 'sitenames':sitenames, 'glon':glon, 'glat':glat,
             'mlon':mlon, 'mlat':mlat, 'mcolat':mcolat, 'BNm':BNm[i], 'BEm':BEm[i], 'BZm':BZm[i],
             'BNg':BNg[i], 'BEg':BEg[i], 'BZg':BZg[i], 'mlt':MLT[i], 'decl':DECL[i], 'sza':SZA[i]}     
    return output

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

def angle_func(x, y):

    """Calculate angle for x and y"""
    
    a = np.rad2deg(np.arctan2(x, y))
    return a

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

def Time2Float(x):

    """Converts datetime to float, so that interpolation/smoothing can be performed"""
    
    if (type(x) == np.ndarray) or (type(x) == list):
        emptyarray = []
        for i in x:
            z = (i - datetime.datetime(1970, 1, 1, 0)).total_seconds()
            emptyarray.append(z)
        emptyarray = np.array([emptyarray])
        return emptyarray[0]
    else:
        return (x - datetime.datetime(1970, 1, 1, 0)).total_seconds()

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

def Float2Time(x):

    """Converts array back to datetime so that it can be plotted with time on the axis"""
    
    if (type(x) == np.ndarray) or (type(x) == list):
        emptyarray = []
        for i in x:
            z = datetime.datetime.utcfromtimestamp(i)
            emptyarray.append(z)
        emptyarray = np.array([emptyarray])
        return emptyarray[0]
    else:
        return datetime.datetime.utcfromtimestamp(x)

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

def calculate_magnetic_latitude(glon, glat, dtime):

    """Calculate latlon using spacepy. Matched with BGS IGRF online calculator
    
    Parameters
    -----------
    glon, glat = geographic longitude and latitude
    dtime = datetime object
    
    Returns
    -----------
    magnetic latitude, magnetic longitude"""
    
    datestring = "%02d-%02d-%02dT12:00:00" % (dtime.year, dtime.month, dtime.day)
    #call with altitude in kilometers and lat/lon in degrees 
    Re=6371.0 #mean Earth radius in kilometers
    alt = 1.    
    #setup the geographic coordinate object with altitude in earth radii 
    cvals = coord.Coords([float((alt/Re+Re))/Re,float(glat),float(glon)], 'GEO', 'sph',['Re','deg','deg'])
    #set time epoch for coordinates:
    cvals.ticks=Ticktock([datestring], 'ISO')
    #return the magnetic coords in the same units as the geographic:
    a = cvals.convert('MAG','sph')
    return a.data[0][1], a.data[0][2]
    
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

def calculate_geographic_latitude(mlon, mlat, dtime):
    
    """Calculate geographic latlon from magnetic coords using spacepy. 

    Parameters
    -----------
    mlon, mlat = magnetic longitude and latitude
    dtime = datetime object
    
    Returns
    -----------
    geographic latitude, geographic longitude"""
    
    datestring = "%02d-%02d-%02dT12:00:00" % (dtime.year, dtime.month, dtime.day)
    #call with altitude in kilometers and lat/lon in degrees 
    Re=6371.0 #mean Earth radius in kilometers
    alt = 1.    
    #setup the geographic coordinate object with altitude in earth radii 
    cvals = coord.Coords([float((alt/Re+Re))/Re,float(mlat),float(mlon)], 'MAG', 'sph',['Re','deg','deg'])
    #set time epoch for coordinates:
    cvals.ticks=Ticktock([datestring], 'ISO')
    #return the magnetic coords in the same units as the geographic:
    a = cvals.convert('GEO','sph')
    return a.data[0][1], a.data[0][2]
    
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

def model_profiles(mystr):

    """Return Quebec or British Columbia 1D resistivity models. 
    From Boteler & Pirjola (1998), 'The complex-image method for calculating the
    magnetic and electric fields produced at the surface of the Earth by the 
    auroral electrojet', DOI: 10.1046/j.1365-246x.1998.00388.x
    
    Parameters
    -----------
    mystr = string. "Q" for Quebec, "BC" for British Columbia

    Returns
    -----------
    resistivities = array of resistivity values in Ohm.m
    thicknesses = array of thicknesses in m"""

    if mystr == "BC":
        resistivities = np.array([500., 150., 20., 300., 100., 10., 1.])
        thicknesses = 1000. * np.array([4., 6., 5., 65., 300., 200.])
        
    elif mystr == "Q":
        resistivities = np.array([20000., 200, 1000, 100, 3])
        thicknesses = 1000. * np.array([15, 10, 125, 200])
    else:
        print("Choose Either 'Q' for Quebec or 'BC' for British Columbia")
        return
    return resistivities, thicknesses

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

def Z_Tensor_1D(resistivities, thicknesses, frequencies):

    """Calculate 1D Z-Tensor for given ground resistivity profile.

    Parameters
    -----------
    resistivities = array or list of resistivity values in Ohm.m

    thicknesses = array or list of thicknesses in m.
        **len(resistivities) must be len(thicknesses) + 1**

    frequencies = array or list of frequencies to get response of
    
    Returns
    -----------
    Z = complex array of Z tensor values
    
    Taken from:
    http://www.digitalearthlab.com/tutorial/tutorial-1d-mt-forward/"""
    
    if len(resistivities) != len(thicknesses) + 1:
        print("Length of inputs incorrect!")
        return 
    
    mu = 4*np.pi*1E-7; #Magnetic Permeability (H/m)
    n = len(resistivities);
    master_Z, master_absZ, master_phase = [], [], []

    for frequency in frequencies:   
        w =  2*np.pi*frequency;       
        impedances = list(range(n));
        #compute basement impedance
        impedances[n-1] = np.sqrt(w*mu*resistivities[n-1]*1j);
       
        for j in range(n-2,-1,-1):
            resistivity = resistivities[j];
            thickness = thicknesses[j];
      
            # 3. Compute apparent resistivity from top layer impedance
            #Step 2. Iterate from bottom layer to top(not the basement) 
            # Step 2.1 Calculate the intrinsic impedance of current layer
            dj = np.sqrt((w * mu * (1.0/resistivity))*1j);
            wj = dj * resistivity;
            # Step 2.2 Calculate Exponential factor from intrinsic impedance
            ej = np.exp(-2*thickness*dj);                     
        
            # Step 2.3 Calculate reflection coeficient using current layer
            #          intrinsic impedance and the below layer impedance
            belowImpedance = impedances[j + 1];
            rj = (wj - belowImpedance)/(wj + belowImpedance);
            re = rj*ej; 
            Zj = wj * ((1 - re)/(1 + re));
            impedances[j] = Zj;    
    
        # Step 3. Compute apparent resistivity from top layer impedance
        Z = impedances[0];
        phase = math.atan2(Z.imag, Z.real)
        master_Z.append(Z)
        master_absZ.append(abs(Z))
        master_phase.append(phase)
        #master_res.append((absZ * absZ)/(mu * w))
    return np.array(master_Z)
    
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
    
def E_Field_1D(bx, by, resistivities, thicknesses, timestep = 60., Z = None, calc_Z = True, pad = True, padnum = 450):

    """Calculate horizontal E-field components given Bx, By, resistivities and thicknesses.
    
    Parameters
    -----------
    bx, by = array of Bx, By timeseries in nT

    resistivities = array or list of resistivity values in Ohm.m

    thicknesses = array or list of thicknesses in m.
        **len(resistivities) must be len(thicknesses) + 1**

    timestep = time between samples (default is 60. for minute sampling)
    
    Z = complex Z-tensor array. If not supplied, Z will be calculated from input
        resistivities and thicknesses
    
    Returns
    -----------
    ext, eyt = arrays of electric field components in mV/km"""
    
    if pad == False:
        new_bx = bx
        new_by = by
    else:
        new_bx = np.concatenate((bx[:padnum], bx, bx[-padnum:][::-1]))
        new_by = np.concatenate((by[:padnum], by, by[-padnum:][::-1]))
    
    mu0 = 4*np.pi * 1e-7
    freq = np.fft.fftfreq(new_bx.size, d = timestep)
    freq[0] = 1e-100

    if calc_Z == True:  # if you need to calculate Z
        Z = Z_Tensor_1D(resistivities, thicknesses, freq)
        
    bx_fft = np.fft.fft(new_bx)
    by_fft = np.fft.fft(new_by)

    exw = Z * by_fft/mu0; 
    eyw = -1 * Z * bx_fft/mu0

    ext = 1e-3 * np.fft.ifft(exw).real
    eyt = 1e-3 * np.fft.ifft(eyw).real

    if pad == False:
        return ext, eyt
    else:
        return ext[padnum:-padnum], eyt[padnum:-padnum]
    
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

def Period(bx, timestep):

    """return period for some input for fft calc"""

    freq = np.fft.fftfreq(new_bx.size, d = timestep)
    freq[0] = 1e-100
    per = 1./freq
    ffft = np.fft.fft(new_bx)
    
    return per, ffft
    
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

class Intermag:

    """Class for reading in INTERMAGNET magnetic data files easily.
    Can clean the data (crudely), merge multiple datafiles and calculate geoelectric fields
        
    Parameters
    -----------
    filename = location of INTERMAGNET file
    clean = set to True to clean data using .clean_data() (default = False)


    Attributes
    -----------    
    name = 3-letter IAGA string (e.g., VAL)
    glat, glon = geographic latitude and longitude
    mlat, mlon = aagcmv2 calculated magnetic latitude and longitude
    data_type = type of data in INTERMAGNET file (e.g., XYZF or HDZF)
    bx, by, bz = numpy arrays of raw data 
    bh = calculated horizontal component
    dbh = rate of change of bh
    timedate = array of datetime objects
    timedate_float = array of datetime floats (i.e., seconds from 1970)
    bad_switch = Boolean. If True, all of the data is bad.
        
    **IF YOU RUN .clean_data()**
    bx_clean, by_clean, bz_clean = cleaned bx, by, bz components
    bh_clean = cleaned horizontal component
    dbh_clean = rate of change of cleaned horizontal component
    good_data = boolean array indicating good or bad data
    
    **IF YOU RUN .calculate_efield(resistivities, thicknesses)**
    ex, ey, eh = calculated horizontal electric field components from cleaned bx, by

    Functions
    -----------    
    .clean_data() = cleans the data in place
    .calculate_efield() = calculates horizontal electric fields in mV/km
    
    Example use
    ----------- 
    >>> # read in first and second days of data, cleaning the data for each day
    >>> day1 = Intermag(filename1, clean = True)
    >>> day2 = Intermag(filename2, clean = True)
    >>> # merge the data
    >>> day1.merge_days(day2)
    >>> # clean the data with different std:
    >>> day1.clean_data(standard_deviations = 3, absolute_diff = 5000)
    >>> # calculate surface electric fields for 1D resistivity profile
    >>> day1.calculate_efield(resistivities, thicknesses)
    
    -----------------------------------------------------------------"""
    
    def __init__(self, filename, clean = False):

        f = open(filename, 'r', errors="replace")
        data = f.readlines()
        f.close()

        for index, value in enumerate(data):
            if ("IAGA CODE" in value) or ("IAGA Code" in value):    # name of site
                i = value.split(" ")
                self.name = [x for x in i if x != ""][2]
            if "Geodetic" in value:     # geographic latitude
                if "Latitude" in value:
                    j = value.split(" ")
                    self.glat = float([x for x in j if x != ''][2])
                if "Longitude" in value: # geographic longitude
                    k = value.split(" ")
                    self.glon = float([x for x in k if x != ''][2])
            if ("Reported" in value) and ("#" not in value):
                l = value.split(" ")    # data type (XYZ or HDZ)
                self.data_type = [x for x in l if x != ''][1]
            if "DATE" in value: # start of data in file
                skiprows = index + 1
                break

        # read in the actual data
        self.bx_days, self.by_days = [], []
        timedate1, bx1, by1, bz1 = [], [], [], []
        for index, value in enumerate(data[skiprows:]):
            try:
                split_line = value.split(" ")
                split_line_no_space = [x for x in split_line if x != ""]

                # get list of datetimes
                dates = split_line_no_space[0]
                year, month, day = int(dates[:4]), int(dates[5:7]), int(dates[8:])
                times = split_line_no_space[1]
                hour, minute, second = int(times[:2]), int(times[3:5]), int(times[6:8])
                timedate1.append(datetime.datetime(year, month, day, hour, minute, second))

                # get bx, by, bz
                bx1.append(float(split_line_no_space[3]))
                by1.append(float(split_line_no_space[4]))
                bz1.append(float(split_line_no_space[5]))
                
            except:
                continue
                
        # convert from HDZ to XYZ if needed
        if self.data_type == "HDZF":
            H, D = np.array(bx1), np.array(by1)
            D = D * (np.pi/180.0) / 60.0 #convert D to radians
            bx1 = np.cos(D)*H
            by1 = np.sin(D)*H

        self.timedate = np.array(timedate1)
        self.bx, self.by, self.bz = np.array(bx1), np.array(by1), np.array(bz1)
        self.bh = np.sqrt(self.bx**2 + self.by**2)
        self.dbh = np.gradient(self.bh)
        self.timedate_float = Time2Float(self.timedate)

        # get magnetic coordinates
        calcmlat, calcmlon = calculate_magnetic_latitude(self.glon, self.glat, self.timedate[0])
        
        self.mlat = calcmlat
        self.mlon = calcmlon
        self.bad_switch = False
        if clean == True:
            self.clean_data()
        
        self.good_data = np.ones(len(self.bx))
        self.bx_clean, self.by_clean, self.bz_clean, self.bh_clean, self.dbh_clean = np.ones(len(self.bx)), np.ones(len(self.bx)), np.ones(len(self.bx)), np.ones(len(self.bx)), np.ones(len(self.bx))
        
        #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
        
    def clean_data(self, standard_deviations = 12, absolute_diff = 10000):
    
        """crude cleaning function
        any points > standard_deviations away from mean
        or any points > absolute_diff from mean == removed
        a simple simple linear interpolation is then performed"""
        
        good_data = np.ones(len(self.bx))   # assume all data good at start
        # now figure out which points to discard
        for component in [self.bx, self.by, self.bz]:
            good_std = np.abs(np.mean(component) - component) < standard_deviations * np.std(component)
            good_abs = np.abs(component - np.mean(component)) < 10000
            good_data = good_data * good_std * good_abs
        
        good_data = good_data.astype(bool)
        self.good_data = good_data
 
        # if all points are bad, tell the user, and give arrays of 0s.
        if (sum(good_data) == 0):
            #print(self.name, " is all bad data!")
            self.bad_switch = True
            len_data = len(self.bx)
            self.bx_clean, self.by_clean, self.bz_clean = np.zeros(len_data), np.zeros(len_data), np.zeros(len_data)
            self.bh_clean, self.dbh_clean = np.zeros(len_data), np.zeros(len_data)
            return
            
        # otherwise, interpolate the data
        self.bx_clean = np.interp(self.timedate_float, self.timedate_float[good_data], self.bx[good_data])
        self.by_clean = np.interp(self.timedate_float, self.timedate_float[good_data], self.by[good_data])
        self.bz_clean = np.interp(self.timedate_float, self.timedate_float[good_data], self.bz[good_data])
        self.bh_clean = np.sqrt(self.bx_clean**2 + self.by_clean**2)
        self.dbh_clean = np.gradient(self.bh_clean)


        if np.max(np.abs(self.dbh_clean)) >= 3000:
            self.bad_switch = True

        self.bx_days.append(self.bx_clean)
        self.by_days.append(self.by_clean)
        #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
     
    def bad_test(self):
    
        if (np.max(np.abs(self.bh_clean)) - np.mean(np.abs(self.bh_clean))) > 10000:
            self.bad_switch = True
        if np.max(self.dbh_clean) > 10000.:
            self.bad_switch = True
            self.reason = 1
            print(self.name, self.reason)            
            
        a = np.max(np.abs(np.gradient(self.bx_clean)))
        b = np.max(np.abs(np.gradient(self.by_clean)))
        if max([a, b]) >= 4000:
            self.bad_switch = True
            self.reason = 2
            print(self.name, self.reason)
            
        #if sum(np.gradient(self.bh_clean) == 0) >= 500:
        #    self.bad_switch = True
        #    self.reason = 3
        #    print(self.name, self.reason)
                        
    def merge_days(self, temp):
    
        """Merge the timeseries of two Intermag class instances
        If your two Intermag instances are day1, day2:
        >>> day1.merge_days(day2)"""
        
        labels = ["bx", "by", "bz", "bh", "dbh", "timedate", "timedate_float", "good_data",
                "bx_clean", "by_clean", "bz_clean", "bh_clean", "dbh_clean"]
        
        for label in labels:
            concat_data = np.concatenate((getattr(self, label), getattr(temp, label)))
            setattr(self, label, concat_data)

        labels2 = ["bx_days", "by_days"]
        for label in labels2:
            concat_data = getattr(self, label) + getattr(temp, label)
            setattr(self, label, concat_data)
        
        
    def calculate_efield(self, resistivities, thicknesses, timestep = 60., Z = None, calc_Z = True):
    
        """Calculate horizontal electric field using 1D resistivity profile in mV/km
        If your Intermag instance is day1:
        >>> Qres, Qthick = model_profiles("Q") # get Quebec resistivity profile
        >>> day1.calculate_efield(Qres, Qthick)"""
        
        
        bx_day_means = np.mean(self.bx_days, axis = 1)
        by_day_means = np.mean(self.by_days, axis = 1)
        maxmin_bx_days = np.max(bx_day_means) - np.min(bx_day_means)
        maxmin_by_days = np.max(by_day_means) - np.min(by_day_means)
        
        if np.max([maxmin_bx_days, maxmin_by_days]) > 1000:
            newbx = self.bx_days[0]
            newby = self.by_days[0]
            print(self.name)
            
            for x,y in zip(self.bx_days[1:], self.by_days[1:]):
                xx = x*(newbx[-1]/x[0])
                newbx = np.concatenate((newbx, xx))
        
                yy = y*(newby[-1]/x[0])
                newby = np.concatenate((newby, yy))
        
            self.ex, self.ey = E_Field_1D(newbx, newby, resistivities, thicknesses, timestep,  Z)
            print(self.name, " HAD TO DO SOME SHADY SHIT")
        else:
            self.ex, self.ey = E_Field_1D(self.bx_clean, self.by_clean, resistivities, thicknesses, timestep,  Z)
        
        self.eh = np.sqrt(self.ex**2 + self.ey**2)

    def calculate_angle(self):
        self.angle = angle_func(self.ex, self.ey)
    
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

def INTERMAGNET_all_sites(start_day, numofdays = 3, quiet = False):

    """Read in all of the INTERMAGNET data for a range of days and calculate maximum Eh

    Suggest reading in at least 3 days to avoid FFT edge problems when calculating Eh

    Parameters
    -----------
    start_day = date of interest (datetime object)
    numofdays = number of days

    Returns
    -----------  
    OUTPUT1: dictionary of all site data with uncleaned bx, by, bz
    OUTPUT2: dictionary of sites that had good enough data to calculate ex, ey"""
        
    # Get the Resistivity profile for calculating E-Fields
    Qres, Qthick = model_profiles("Q")
    freq = np.fft.fftfreq(numofdays * 1440, d = 60.)
    freq[0] = 1e-100
    ZZ = Z_Tensor_1D(Qres, Qthick, freq)

    wanted_days = [start_day + datetime.timedelta(days = x) for x in range(numofdays)]

    # First, get the names of all of the available data files for the three days
    all_data_files = []
    for i in wanted_days:

        if quiet == False:
            print(i)

        chosen_year = i.year
        chosen_month = i.month
        chosen_day = i.day

        chosen_folder = INTERMAGdatafolder + "%02d/%02d/" % (chosen_year, chosen_month)
        datestring = "%02d%02d%02d" % (chosen_year, chosen_month, chosen_day)
        
        fnames = os.listdir(chosen_folder)
        all_data_files.extend([chosen_folder + x for x in fnames if datestring in x])    
        
    # Now read in the data into Intermag class objects
    # Get the names of all of the sites first
    sitenames = sorted(list(set([x.split("/")[-1][:3] for x in all_data_files])))

    master_data, master_data2 = {}, {}
    for sitename in sitenames:
        chosen_filenames = sorted([x for x in all_data_files if sitename in x])
        
        # Only want to look at sites with 3 datafiles
        if len(chosen_filenames) < 3:
            continue

        # If there are multiple files, read them in and append   
        data = Intermag(chosen_filenames[0], clean = False)

        for jjj in chosen_filenames[1:]:        
            temp1 = Intermag(jjj, clean = False)
            data.merge_days(temp1)
        
        data.clean_data()
        master_data[data.name] = data

    # Now loop through the sites. If they are 'clean', calculate E-fields, and add
    # them to our master_data2 dictionary.
    for i in master_data:
        master_data[i].bad_test()
        if master_data[i].bad_switch == True:
            continue
        else:
            master_data[i].calculate_efield(Qres, Qthick, Z = ZZ)
            master_data2[i] = master_data[i]


    OUTPUT1, OUTPUT2 = {}, {}
    for i in master_data:
        XXX = master_data[i]
        OUTPUT1[i] = {'glon':XXX.glon, 'glat':XXX.glat, 'mlon':XXX.mlon, 'mlat':XXX.mlat, 'timedate':XXX.timedate,
                    'bx':XXX.bx, 'by':XXX.by, 'bz':XXX.bz, 'bh':XXX.bh}

    for i in master_data2:
        XXX = master_data2[i]
        OUTPUT2[i] = {'glon':XXX.glon, 'glat':XXX.glat, 'mlon':XXX.mlon, 'mlat':XXX.mlat, 'timedate':XXX.timedate,
                    'bx':XXX.bx_clean, 'by':XXX.by_clean, 'bz':XXX.bz_clean, 'bh':XXX.bh_clean,
                    'ex':XXX.ex, 'ey':XXX.ey, 'eh':XXX.eh}
                    
    return OUTPUT1, OUTPUT2
    
    
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
    
def INTERMAGNET_single_site(site, start_day, numofdays = 3, quiet = False):

    """For a given site, read in multiple days of INTERMAGNET data
    
    Parameters
    -----------
    site = IAGA string (e.g., 'VAL')
    start_day = beginning date of interest
    numofdays = number of days needed
    quiet = print output (false by default)

    Returns
    -----------  
    data:  Intermag object of data with calculated Eh"""
    
    # Get the Resistivity profile for calculating E-Fields
    Qres, Qthick = model_profiles("Q")
    freq = np.fft.fftfreq(4320, d = 60.)
    freq[0] = 1e-100
    ZZ = Z_Tensor_1D(Qres, Qthick, freq)
    
    wanted_days = [start_day + datetime.timedelta(days = x) for x in range(numofdays)]

    # First, get the names of all of the available data files for the three days
    all_data_files = []
    for i in wanted_days:
        if quiet == False:
            print(i)
            
        chosen_year = i.year
        chosen_month = i.month
        chosen_day = i.day

        chosen_folder = INTERMAGdatafolder + "%02d/%02d/" % (chosen_year, chosen_month)
        datestring = "%02d%02d%02d" % (chosen_year, chosen_month, chosen_day)
        
        fnames = os.listdir(chosen_folder)
        all_data_files.extend([chosen_folder + x for x in fnames if datestring in x]) 

    wanted_data_files = [x for x in all_data_files if site.lower() in x]
    if len(wanted_data_files) == 0:
        print("NO DATA")
        return
        
    data = Intermag(wanted_data_files[0], clean = False)
    for i in wanted_data_files[1:]:
        temp1 = Intermag(i, clean = False)
        data.merge_days(temp1)
        
    data.clean_data()

    if data.bad_switch == True:
        print("BField data bad, manually calculate EField")
    else:
        data.calculate_efield(Qres, Qthick, Z = ZZ)

    return data
    
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

def get_Dst_values(timerange = 'short', sort_switch = False, bigtimeseries = False):

    """Return Dst values from http://wdc.kugi.kyoto-u.ac.jp/wdc/Sec3.html
    
    Parameters
    -----------
    timerange = 'short' for 1991-2019, 'long' for 1957-2019
    sort_switch = set to True to order by minimum Dst, False by default
    bigtimeseries = set to True to return 3-hourly bins, False by default
                    this will return timedate and Dst_values only

    Returns
    -----------  
    by default:
    timedate = array of days (datetime object)
    Dst_values = array of 3-hourly values for each day
    Dst_mins = array of daily minimum Dst
    
    if bigtimeseries = True:
    timedate = array of hours (datetime object)
    Dst_values = hourly Dst values"""

    if (timerange=="long") or (timerange=="LONG"):
        filename = DSTfolder + 'DST_VALUES_1957_2019.txt'
    else:
        filename = DSTfolder + 'DST_VALUES_1991_2019.txt'
        
    data = np.loadtxt(filename)

    DST_timedate = Float2Time(data[:,0])
    DST_values = data[:,1:]
    DST_values_min = np.min(DST_values, axis = 1)

    if (sort_switch == True):   # order by daily minimum Dst
        sort_ind = np.argsort(DST_values_min)
        return DST_timedate[sort_ind], DST_values[sort_ind], DST_values_min[sort_ind]
        
    if (bigtimeseries == True): # return 3-hour bins
        newtd, newvalues = [], []
        for i1, v1 in enumerate(DST_timedate):
            for i in range(24):
                newtime = v1 + datetime.timedelta(minutes = 60*i)
                newtd.append(newtime)
                newvalues.append(DST_values[i1][i])
                
        return np.array(newtd), np.array(newvalues)
    else:
        return np.array(DST_timedate), np.array(DST_values), np.array(DST_values_min)
        
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

def get_SYMH_values():

    """Return SYMH values for 1991-2018
    warning, this is pretty slow (big time-series)
    
    Returns
    -----------  
    timedate = array of days (datetime object)
    SYMH_values = SYMH values nT"""

    data = np.load(DSTfolder + "SYMH_VALUES_1991_2018.npy")
    
    SYMH_timedate = Float2Time(data[:,0])
    SYMH_values = data[:,1]
    
    return(SYMH_timedate, SYMH_values)

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

def get_mlat_EH(md2):

    """Get mlat and max Eh from collection of Intermag objects
    
    Parameters
    -----------  
    md2 = dictionary of Intermag values
    
    Returns
    -----------  
    MLATS = array of magnetic latitudes
    EH = array of maximum EH values"""    
    
    MLATS, EH = [], []
    for i in md2:
        site = md2[i]
        MLATS.append(site['mlat'])
        EH.append(np.max(site['eh'][50:-50]))
    
    MLATS, EH = np.array(MLATS), np.array(EH)
    ind = MLATS.argsort()
    
    return MLATS[ind], EH[ind]
    
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

def fit_spline_lat_e(lats, maxe, bootstrap_num = 500, lat_cutoff = 10, uselog = True):

    """Calculate Spline Bootstrap fit for input lats and max Eh. Calculates using
    random subselection bootstrap_num times, then calculates using all data
    
    inputs:
    -------------------------------------------------------------
    lats = magnetic latitude of sites in deg
    maxe = maximum calculated Eh at sites (usually V/km)
    bootstrap_num = number of times to calculate boundary with random 
                    subselection of points
    lat_cutoff = latitude South of which points are ignored for fitting
    uselog = whether to use log10 of maxe or not for fitting 
    
    outputs:
    -------------------------------------------------------------
    xthresh = all of the latitude threshold points calculated from bootstrap
    ythresh = all of the E values for each threshold point
    gradients = gradients at each point
    absfit = np.array([xx1, abs_yy1]) = the curve calculated with 100% of the points
    """

    X = lats
    Y = maxe
    ind = X >= lat_cutoff

    # add a very small number to each mlat element
    # auroral boundary algo needs all mlat values to be unique    
    if len(np.unique(X)) != len(X):
        incr = np.linspace(0.000001, 0.000005, len(X))
        X = X + incr 

    if uselog == True:
        x1, y1 = X[ind], np.log10(Y[ind])
    else:
        x1, y1 = X[ind], Y[ind]
        
    # This is the smoothing factor
    p = 400
    
    # Latitude points to calculate fit 
    xx1 = np.linspace(x1[0], x1[-1], 1000)

    # calculate the fit for 75% of the points, bootstrap_num times
    xthresh, ythresh, gradients = [], [], []
    while len(xthresh) < bootstrap_num:
        # Randomly select 75% of the points
        indices = np.random.choice(len(x1), int(len(x1)*(0.75)), replace=False)
        x1_new, y1_new = x1[indices], y1[indices]
        x1_new, y1_new = zip(*sorted(zip(x1_new, y1_new)))
        x1_new, y1_new = np.array(x1_new) + (np.arange(len(x1_new)) / 1e7), np.array(y1_new)

        # Calculate spline fit with this reduced set
        xx1 = np.linspace(x1[0], x1[-1], 1000)
        w1 = np.ones_like(y1_new)
        GCV_manual = SmoothedNSpline(x1_new, y1_new, w=w1, p=p)
        yy1 = GCV_manual(xx1)
        gradyy1 = np.gradient(yy1)
        xgradyy1 = xx1[gradyy1.argmax()]

        # If the fit misattributes it to a point below the lat_cutoff, ignore
        if xgradyy1 >= lat_cutoff:
            gradients.append(np.max(gradyy1))
            xthresh.append(xgradyy1)
            ythresh.append(yy1[gradyy1.argmax()])


    # Now calculate with all of the data
    w1 = np.ones_like(y1)
    GCV_manual2 = SmoothedNSpline(x1, y1, w=w1, p=p)
    abs_yy1 = GCV_manual2(xx1)
    abs_gradyy1 = np.gradient(abs_yy1)
    abs_xgradyy1 = xx1[abs_gradyy1.argmax()]
     
    return xthresh, ythresh, gradients, np.array([xx1, abs_yy1])

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-


























































































    
    
