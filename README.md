# G-MILOS

## Description 

Authors: Manuel Cabrera, Juan P. Cobos, Luis Bellot Rubio (IAA-CSIC).

This work has received funding from the European Union's Horizon 2020 research and innovation programme under grant agreement No 824135 (SOLARNET).

For questions, please contact Luis Bellot (lbellot@iaa.es).





## Introduction

This repository contains G-MLOS, a CUDA implementation of the P-MILOS inversion code. G-MILOS is the first Milne-Eddington code based on the Levenberg-Marquardt algorimth running on Graphics Processing Units (GPUs). It can invert full spectropolarimetric measurements of photospheric spectral lines using a one-component Milne-Eddington atmosphere and taking into account the transmission profile of the instrument and stray-light contamination. The code is very fast, reaching speeds of 7400 pixels per second on an NVIDIA Tesla V100 GPU. This refers to the inversion of a full Stokes data cube in FITS format (4 Stokes profiles, 30 wavelength samples) with 9 free parameters, a maximum of 50 iteration steps, and PSF convolution. 

In what follows we explain how to install and run the code. We also provide a brief overview of the input/output files. A complete user manual can be found [here](g-milos_manual.pdf).


## Requeriments 

Both C and CUDA must be installed on the system. The oldest CUDA version supported by G-MILOS is 3.5, but we strongly recommend you to use the latest versions of the CUDA Toolkit and the  Intel C compiler to achieve maximum performance. 


### Libraries

The following libraries are needed: 

- [CFITSIO](https://heasarc.gsfc.nasa.gov/fitsio/) (Oldest version tested 3.3.4.0)
- [GSL](https://www.gnu.org/software/gsl/) (Oldest version tested 1.13-3)
  
There are different ways to install them depending on the operating system. On Ubuntu you can use the following commands:

CFITSIO

```
sudo apt-get update -y 
sudo apt-get install libcfitsio*
```


GSL

```
sudo apt-get update -y 
sudo apt-get install libgsl*
```

## Compilation

The code needs to be compiled on the target machine. To do that, run the command 'make' in the directory where the source code is located. 

There are two environment variables you must define in the shell, namely CUDA_PATH and SMS. The first one contains the path of the CUDA Toolkit and the second gives the CUDA compatibility with which the code will be compiled (this value must be known from the graphics card architecture). Here is an example how to create the two environment variables using a bash command console. To make these variables permanent, add them to your ~/.bashrc file.

```
export CUDA_PATH="/usr/local/cuda-10.1"
export SMS="35"
```

If these variables are not defined, the makefile will use CUDA_PATH=“/usr/local/cuda-10.1” and will compile the code for GPUs with compute capability 35 37 50 52 60 and 70. 

* Compile and create executable **gmilos** 
```
make 
```
* Clean object files and executable files. 
```
make clean
```

## Execution

G-MILOS uses an ASCII control file with extension **.mtrol**. An example ([invert.mtrol](run/invert.mtrol))can be found in the run directory. Please refer to the user manual for a detailed explanation of the different parameters in the control file. 

The code is executed by passing the control file as a parameter. There are two examples in this repository. The first one executes a spectral synthesis for the given model atmosphere:

```
./gmilos run/synthesis.mtrol
```

The second example inverts the profiles generated in the previous synthesis:

```
./gmilos run/invert.mtrol
```

In both cases, the results are stored in the directory **data**. 


## Input/output files

### Profile files (.per)

The Stokes profiles of an individual pixel can be stored in an ASCII file with extension **.per**. These have the same format as SIR .per files. They are used as input when inverting one pixel and as output when synthesizing the profiles from a given model atmosphere.

Profile files have one row per wavelength sample and 6 columns containing:

* The index of the spectral line in the atomic parameter file
* The wavelength offset with respect to the central wavelength (in mA) 
* The value of Stokes I
* The value of Stokes Q
* The value of Stokes U
* The value of Stokes V

This is an example of a file containing the Stokes parameters of spectral line number 1 in 30 wavelength positions, from -350 to + 665 mA:

```
1    -350.000000    9.836711e-01    6.600326e-04    4.649822e-04    -3.694108e-03
1    -315.000000    9.762496e-01    1.186279e-03    8.329745e-04    -6.497371e-03
1    -280.000000    9.651449e-01    2.305113e-03    1.581940e-03    -1.135160e-02
1    -245.000000    9.443904e-01    5.032997e-03    3.333831e-03    -2.191048e-02
1    -210.000000    9.018359e-01    1.146227e-02    7.265856e-03    -4.544966e-02
1    -175.000000    8.222064e-01    2.265146e-02    1.368713e-02    -8.623441e-02
1    -140.000000    7.066048e-01    3.263217e-02    1.847884e-02    -1.242511e-01
1    -105.000000    5.799722e-01    3.157282e-02    1.560218e-02    -1.238010e-01
1    -70.000000    4.711627e-01    2.068015e-02    6.887295e-03    -8.728459e-02
1    -35.000000    4.014441e-01    9.837587e-03    -1.054865e-03    -4.476189e-02
1    -0.000000    3.727264e-01    4.631597e-03    -4.830483e-03    -9.482273e-03
1    35.000000    3.799767e-01    5.985593e-03    -3.846331e-03    2.321622e-02
1    70.000000    4.249082e-01    1.378049e-02    1.806246e-03    6.157872e-02
1    105.000000    5.119950e-01    2.571598e-02    1.073034e-02    1.046772e-01
1    140.000000    6.316050e-01    3.361904e-02    1.782490e-02    1.297000e-01
1    175.000000    7.571660e-01    2.941514e-02    1.718121e-02    1.115998e-01
1    210.000000    8.600360e-01    1.762856e-02    1.087526e-02    6.786425e-02
1    245.000000    9.230015e-01    8.213106e-03    5.305210e-03    3.366419e-02
1    280.000000    9.545796e-01    3.605521e-03    2.427676e-03    1.654710e-02
1    315.000000    9.701367e-01    1.734934e-03    1.205792e-03    9.068003e-03
1    350.000000    9.786569e-01    9.418463e-04    6.697733e-04    5.552034e-03
1    385.000000    9.838719e-01    5.600237e-04    4.053978e-04    3.671347e-03
1    420.000000    9.873039e-01    3.608273e-04    2.646724e-04    2.540035e-03
1    455.000000    9.896545e-01    2.543154e-04    1.872806e-04    1.792397e-03
1    490.000000    9.912774e-01    1.968866e-04    1.437179e-04    1.263287e-03
1    525.000000    9.923597e-01    1.690016e-04    1.205914e-04    8.552026e-04
1    560.000000    9.929766e-01    1.638180e-04    1.128983e-04    4.949613e-04
1    595.000000    9.930763e-01    1.826333e-04    1.215294e-04    1.047099e-04
1    630.000000    9.923094e-01    2.385952e-04    1.569032e-04    -4.666936e-04
1    665.000000    9.895550e-01    3.734447e-04    2.531845e-04    -1.597078e-03
```

### Profile files (.fits) 

G-MILOS can invert single data cubes containing the Stokes profiles observed in the entire field of view. 

The data cubes must be written as a 4-dimension array in FITS format, with one cube containing one spectral scan. The four dimensions correspond to the wavelength axis, the polarization axis, and the two spatial coordinates x and y. The number of elements in each dimension is n_lambdas, n_stokes, n_x, and n_y, respectively. The cube can be arranged in any other order, but the FITS header must contain the keywords CTYPE1, CTYPE2, CTYPE3 and CTYPE4 to specify each dimension according to the SOLARNET standard:

- **HPLN-TAN** indicates a spatial coordinate axis  
- **HPLT-TAN** indicates a spatial coordinate axis
- **WAVE-GRI** indicates the wavelength axis
- **STOKES  '** indicates the Stokes parameter axis

The example below corresponds to a data cube with the wavelength grid in the first dimension, the polarization axis in the second dimension, the x-spatial coordinate in the third dimension, and the y-spatial coordinate in the fourth dimension. 

```
CTYPE1  = 'WAVE-GRI' 
CTYPE2  = 'STOKES  ' 
CTYPE3  = 'HPLN-TAN'
CTYPE4  = 'HPLT-TAN  ' 
```

When the FITS data cube does not have a header, the array is assumed to be ordered as (n_lambdas, n_stokes, n_x, n_y).


### Wavelength grid file (.grid)

The wavelength grid file specifies the spectral line and the observed wavelength positions  (inversion mode) or the wavelength positions in which the profiles must be calculated (synthesis mode).  The line is identified by means of an index that must be present in the atomic parameter file. The wavelength range is given using three numbers: the initial wavelength, the wavelength step, and the final wavelength (all in mA). 

This file is written in ASCII and has the same format as the **.grid** SIR files. Here is an example:

```
Line indices            :   Initial lambda     Step     Final lambda
(in this order)                    (mA)          (mA)         (mA) 
-----------------------------------------------------------------------
1                       :        -350,            35,           665
```
This example corresponds to the file [malla.grid](run/malla.grid) in the *run* directory.


### Wavelength grid file (.fits)

The wavelenght positions can also be given in FITS format. If all pixels use the same wavelength grid, the FITS file should contain a 2-dimension array with (1, n_lambdas) elements. The first dimension contains the index of the line in the atomic parameter file and the second dimension the observed wavelengths. 


### Model atmosphere file (.mod)

Files with extension **.mod** are ASCII files containing the parameters of a Milne-Eddington model atmosphere. They are used in three situations:

1. To specify the model atmosphere in a spectral synthesis
2. To specify the initial model atmosphere in an inversion 
3. To store the best-fit model atmosphere resulting from the inversion of a profile provided as a **.per** file. 

The following is an example of a model atmosphere file that can be used for the Fe I 6173 A line in the quiet Sun:

```
eta_0                :13.0
magnetic field [G]   :500.
LOS velocity [km/s]  :0.2
Doppler width [A]    :0.035
damping              :0.19
gamma [deg]          :30.
phi   [deg]          :30.
S_0                  :0.26
S_1                  :0.74
v_mac [km/s]         :0.
filling factor       :1
```

This file is different from the equivalent SIR file because Milne-Eddington atmospheres can be described with only 11 parameters. The units of the parameters are: Gauss (magnetic field strength), km/s (LOS velocity and macroturbulent velocity v_mac), Angstrom (Doppler width), and degrees (inclination gamma and azimuth phi). The rest of parameters do not have units. 

### Model atmosphere file (.fits) 

When full data cubes are inverted, the resulting model atmospheres are stored in FITS format as 3-dimension arrays with (n_x, n_y, 13) elements. The first two dimensions give the spatial coordinates x and y. The third dimension contains the eleven parameters of the model, plus the number of interations used by the code to find the solution and the chisqr-value of the fit. Therefore, the 13 values stored in the third dimension are: 

1. eta0 = line-to-continuum absorption coefficient ratio         
2. B = magnetic field strength       [Gauss]
3. vlos = line-of-sight velocity       [km/s]         
4. dopp = Doppler width               [Angstroms]
5. aa = damping parameter
6. gm = magnetic field inclination [deg]
7. az = magnetic field azimuth      [deg]
8. S0 = source function constant  
9. S1 = source function gradient
10. mac = macroturbulent velocity  [km/s]
11. alpha = filling factor of the magnetic component [0->1]
12. Number of iterations required 
13. chisqr value of the fit

The file name of the output models is constructed from the name of the Stokes profiles cube, adding the string '_mod.fits'.

