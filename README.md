# GPU-MILOS

## Description 

This repository contains an implementacion of MILOS using CUDA and will get you a copy of the project up and running on your local machine for development and testing purposes. An extended user manual can be found [here](gmilos_manual.pdf). But in this page you can find a quick overview about how to install the necessary libraries, the types of files used and how to use the program. In this manual we will assume that CUDA is installed on the system. The oldest version of CUDA supported by the code is 3.5 . 


## Requeriments 

### Libraries

The following libraries and tools must be installed in your system: 

- [CFITSIO](https://heasarc.gsfc.nasa.gov/fitsio/) (Minor version tested 3.3.4.0)
- [GSL](https://www.gnu.org/software/gsl/) (Minor version tested 1.13-3)
  
There are many differents ways to install them depending of OS what we are using. In our case we have been using Ubuntu 18.04 as OS, and these are the command to install each library, if you are in the same situation. For other OS, it's in your hands install the specific libraries.


CFITSIO:

```
sudo apt-get update -y 
sudo apt-get install libcfitsio*
```


GSL:

```
sudo apt-get update -y 
sudo apt-get install libgsl*
```

### Files format

#### .per

It's used to specify one profile. We will use it and input for inversion one pixel and as output for synthesis of one model.
It contains 6 columns:

* The first is the index of the spectral line used in the spectral lines file.
* The second is the offset of wavelenght respect the central wavelenght. 
* Value of I
* Value of Q
* Value of U
* Value of V

This is an example of one line: 

```
1	-0.350000	8.480462e-01	2.081567e-05	-3.810591e-05	-2.589682e-04
```


#### .fits 

* Spectro 

The **fits** files used for pass to the program the spectro image must contain four dimensions: *number_rows*X*number_cols*number_of_wavelengths*X*number_stokes*X* . The order or these parameters cannot change and for identify each one the header of **fits** file must contain the type of each dimension with this correspondence:

  - Number of Rows: include CTYPE with the value **'HPLN-TAN'**
  - Number of Cols: include CTYPE with the value **'HPLT-TAN'**
  - Number of Wavelenghts: include CTYPE with the value **'WAVE-GRI'**
  - Number of Stokes: include CTYPE with the value **'STOKES  '**

An example can be this:

```
CTYPE1  = 'HPLN-TAN' 
CTYPE2  = 'HPLT-TAN' 
CTYPE3  = 'WAVE-GRI'
CTYPE4  = 'STOKES  ' 
```

* Wavelengths

If the observed spectra of all pixels use the same wavelength grid, the FITS file must contain a single, 2D array with dimension number of wavelength-points×2. The first column must contain the index with which the spectral line is identified according to the atomic parameter file.

* Output Models 

For save the output models of invert one image, the program use FITS. The data is saved in FLOAT precision and the dimensiones of image will be: numberOfRows X numberOfCols X 13. The number 13 comes from the eleven parameters of the model, the number of interations used by the algorithm to found the solution in that pixel and the value of Chisqr calculated for the result model of that pixel respect the input profile. Therefor, the order of the third dimension of the file will be: 

  1. eta0 = line-to-continuum absorption coefficient ratio         
  2. B = magnetic field strength       [Gauss]
  3. vlos = line-of-sight velocity     [km/s]         
  4. dopp = Doppler width              [Angstroms]
  5. aa = damping parameter
  6. gm = magnetic field inclination   [deg]
  7. az = magnetic field azimuth       [deg]
  8. S0 = source function constant
  9. S1 = source function gradient
  10. mac = macroturbulent velocity     [km/s]
  11. alpha = filling factor of the magnetic component [0->1]
  12. Number of iterations needed. 
  13. Value of Chisqr. 

#### .grid

This is the file where you can specify the number of line from your file with spectral lines to use and the range of wavelenghts to use. This range will be specify with an initial lambda, a step between each wavelenght and the final lambda of the range. 

Look this example:

```
Line indices            :   Initial lambda     Step     Final lambda
(in this order)                    (mA)          (mA)         (mA) 
-----------------------------------------------------------------------
1                       :        -350            35           665
```
In the file [malla.grid](run/malla.grid) you can find an extended example. 


#### .mod 

These files will be used for three purposes:

  1. For specify the initial model of a synthesis.  
  2. For specify the initial model of a inversion. 
  3. For save output model when we are doing the inversion of a profile stored in a .per file. 

The order of parameters in the file must be always the same. This is an example: 

```
eta_0:          14
magnetic field: 1200
LOS velocity:   0
Doppler width:  0.07
damping:        0.05
gamma:          130
phi:            25
S_0:            0.25
S_1:            0.75
v_mac:          1
filling factor: 1
```


## Instalation

In order to deploy the application, it must first be compiled on the target machine. To do this, you must use the command line option 'make' from same directory where the source code is located. So, the first thing is to position ourselves in the GPU-MILOS. After that you must edit the file "makefile" and edit variable CUDA_PATH with the location of CUDA Toolkit in your machine. By default the makefile compile the code with compatibility for this generations of CUDA: 35 37 50 52 60 70 , if you want compile for a specific generation edit file "makefile", searh variable SMS and modify it with the value you want. 


* Compile and create executable **gmilos** 
```
make 
```
* Clean objects files and executable files. 
```
make clean
```

## Deployment


### gmilos

The program must be controlled with a configuration file of type **.mtrol** . Inside the run folder, you can find an example of this type of file [gmilos.mtrol](run/gmilos.mtrol). We refer you to the pdf documentation to know in detail how each parameter works. 

The program must be executed by passing the configuration file as a parameter:

```
./gmilos run/gmilos.mtrol
```
