import numpy as np
import matplotlib.pyplot as plt
import datetime


# Assorted codes










###############################################################################

if 0:
    # Read in ground magnetic data from Intermagnet and Supermag for a particular day:
    # interested in the halloween 2003 storms
    start_day = datetime.datetime(2003, 10, 29)
    numofdays = 5
    end_day = start_day + datetime.timedelta(days=numofdays)

    # Let's read in the Eskdalemuir site only.
    # this is an Intermag object which tries to clean the data, then calculates E
    ESKdata = INTERMAGNET_single_site('ESK', start_day, numofdays = 5, quiet = True)
    bx, by = ESKdata.bx_clean, ESKdata.by_clean
    ex, ey = ESKdata.ex, ESKdata.ey

    # If we were inclined to calculate 1D E-field for a particular profile manually:
    Qres, Qthick = model_profiles("Q")  # use Quebec 1D profile
    ex_man, ey_man = E_Field_1D(bx, by, Qres, Qthick, calc_Z = True)

    fig1 = plt.figure(1)
    plt.clf()

    ax1 = plt.subplot(211)
    plt.title("Fig1: ESK Intermag only", fontsize = 20)
    plt.plot(ESKdata.timedate, ESKdata.bx_clean - ESKdata.bx_clean[0], label = 'bx')
    plt.plot(ESKdata.timedate, ESKdata.by_clean - ESKdata.by_clean[0], label = 'by')
    plt.legend()
    plt.grid(True)

    ax2 = plt.subplot(212)
    plt.plot(ESKdata.timedate, ESKdata.ex, label = 'ex')
    plt.plot(ESKdata.timedate, ESKdata.ey, label = 'ey')
    plt.legend()
    plt.grid(True)

    plt.show()
        
###############################################################################

if 0:
    # Now let's look at all of the available sites for the same period. Intermagnet first:
    # Imag1 is a dictionary of all of the available data uncleaned
    # Imag2 is a dictionary of the 'good' enough cleaned data, with E-values
    Imag1, Imag2 = INTERMAGNET_all_sites(start_day, numofdays = 5, quiet = True)

    # Now we can get the mlat and max Eh for the Intermagnet data:
    Imag_mlat, Imag_Eh = get_mlat_EH(Imag2)

    # And lets fit a spline-fit to this
    x1, y1, g1, fit1 = fit_spline_lat_e(Imag_mlat, Imag_Eh)

    # And now lets read in all of the SuperMag data and calculate E:
    # This takes a few seconds, depending on the number of Supermag sites
    Smag1 = SuperMag_data_local(start_day, end_day, calc_efield = True)
    Smag_mlat, Smag_Eh = get_mlat_EH(Smag1)
    x2, y2, g2, fit2 = fit_spline_lat_e(Smag_mlat, Smag_Eh)
        
    # note, SuperMag seems to apply a cleaning algorithm to their magnetic data
    # which results in missing values compared to Intermagnet?
        
    fig2 = plt.figure(2)
    plt.clf()

    ax1 = plt.subplot(211)
    plt.scatter(Imag_mlat, Imag_Eh)
    plt.plot(fit1[0], 10**fit1[1])
    plt.grid(True)
    plt.title("Fig2: Intermag sites + fit", fontsize = 16)
    plt.ylabel("Eh (mV/km)", fontsize = 20)

    ax2 = plt.subplot(212)
    plt.scatter(Smag_mlat, Smag_Eh)
    plt.plot(fit2[0], 10**fit2[1])
    plt.grid(True)
    plt.title("Supermag sites + fit", fontsize = 16)
    plt.ylabel("Eh (mV/km)", fontsize = 20)
    plt.xlabel("MLAT", fontsize = 20)
    
    plt.show()

###############################################################################

if 0:
    # read in some Dst and SYMH which are saved locally:
    #Dsttd, Dstvals, Dstmins = get_Dst_values()
    #Dsttd2, Dstvals2, Dstmins2 = get_Dst_values(timerange='long')
    Dsttdbig, Dstvalsbig = get_Dst_values(bigtimeseries = True) # to get hourly dst

    SYMHtd, SYMHvals = get_SYMH_values()    # read in SYMH values

    fig3 = plt.figure(3)
    plt.clf()

    plt.title("Fig3: Dst and SYMH", fontsize = 20)
    plt.plot(Dsttdbig, Dstvalsbig, label = 'DST')
    plt.plot(SYMHtd, SYMHvals, label = 'SYMH')
    plt.legend()
    
    plt.grid(True)
    plt.xlim([start_day, end_day])
    plt.ylabel("nT", fontsize = 20)
    plt.show()

###############################################################################
# Now for SWMF stuff
###############################################################################

if 0:
    # read in and quickly plot solar wind file:
    filename = 'INPUT_DATA/SCEN1.dat'
    SWdata = solar_wind_reader(filename, return_data = True, makeplot = True)

###############################################################################

if 0:
    # Now read in all of the surface mag fields in the SWMF GM/IO2/ folder, and save
    # as numpy objects for each magnetic variable needed (bx, mx, px etc.)
    # I do this as reading in 12 numpy array objects is quicker than reading in 
    # hundreds of SWMF mag_grid files
    infold = 'INPUT_DATA/'
    inputcols = ['Bx', 'By', 'Bz', 'Mx']
    outfold = 'OUTPUT_DATA/'
    outprefix = 'TEST_'
    SWMFmagdata1 = read_in_raw_SWMF_magfiles(infold, inputcols, outfold=outfold, outprefix=outprefix)

    # if we come back and want to quickly read in this saved data, we can point to
    # the folder of saved numpy files and do the following:
    SWMFmagdata2 = read_in_SWMF_magfiles(outfold, prefix = outprefix)

###############################################################################

if 0:
    # Calculate magnteopause standoff distance from BATSRUS y=0*.out files
    folder = 'INPUT_DATA/'
    imagefolder = 'OUTPUT_IMAGES/'
    Mpause_td, Mpause, fig = get_magpause_distance(folder, makeplot = True, imagefolder = imagefolder)

###############################################################################

if 1:
    # make plane plots of the azimuthal current from a 3d*.out file
    filename = 'INPUT_DATA/SCEN1_3_0630_3d.out'
    imagefolder = 'OUTPUT_IMAGES/'
    slices = [-3, -1, -0.5, 0, 0.5, 1, 3] # Re for XY and XZ planes 
    Jaz_plane_plotter(filename, imagefolder, slices = slices)

















    
    
    
    
    
    

