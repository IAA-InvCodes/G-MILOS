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
1	-350.000000	9.836711e-01	6.600326e-04	4.649822e-04	-3.694108e-03
1	-315.000000	9.762496e-01	1.186279e-03	8.329745e-04	-6.497371e-03
1	-280.000000	9.651449e-01	2.305113e-03	1.581940e-03	-1.135160e-02
1	-245.000000	9.443904e-01	5.032997e-03	3.333831e-03	-2.191048e-02
1	-210.000000	9.018359e-01	1.146227e-02	7.265856e-03	-4.544966e-02
1	-175.000000	8.222064e-01	2.265146e-02	1.368713e-02	-8.623441e-02
1	-140.000000	7.066048e-01	3.263217e-02	1.847884e-02	-1.242511e-01
1	-105.000000	5.799722e-01	3.157282e-02	1.560218e-02	-1.238010e-01
1	-70.000000	4.711627e-01	2.068015e-02	6.887295e-03	-8.728459e-02
1	-35.000000	4.014441e-01	9.837587e-03	-1.054865e-03	-4.476189e-02
1	-0.000000	3.727264e-01	4.631597e-03	-4.830483e-03	-9.482273e-03
1	35.000000	3.799767e-01	5.985593e-03	-3.846331e-03	2.321622e-02
1	70.000000	4.249082e-01	1.378049e-02	1.806246e-03	6.157872e-02
1	105.000000	5.119950e-01	2.571598e-02	1.073034e-02	1.046772e-01
1	140.000000	6.316050e-01	3.361904e-02	1.782490e-02	1.297000e-01
1	175.000000	7.571660e-01	2.941514e-02	1.718121e-02	1.115998e-01
1	210.000000	8.600360e-01	1.762856e-02	1.087526e-02	6.786425e-02
1	245.000000	9.230015e-01	8.213106e-03	5.305210e-03	3.366419e-02
1	280.000000	9.545796e-01	3.605521e-03	2.427676e-03	1.654710e-02
1	315.000000	9.701367e-01	1.734934e-03	1.205792e-03	9.068003e-03
1	350.000000	9.786569e-01	9.418463e-04	6.697733e-04	5.552034e-03
1	385.000000	9.838719e-01	5.600237e-04	4.053978e-04	3.671347e-03
1	420.000000	9.873039e-01	3.608273e-04	2.646724e-04	2.540035e-03
1	455.000000	9.896545e-01	2.543154e-04	1.872806e-04	1.792397e-03
1	490.000000	9.912774e-01	1.968866e-04	1.437179e-04	1.263287e-03
1	525.000000	9.923597e-01	1.690016e-04	1.205914e-04	8.552026e-04
1	560.000000	9.929766e-01	1.638180e-04	1.128983e-04	4.949613e-04
1	595.000000	9.930763e-01	1.826333e-04	1.215294e-04	1.047099e-04
1	630.000000	9.923094e-01	2.385952e-04	1.569032e-04	-4.666936e-04
1	665.000000	9.895550e-01	3.734447e-04	2.531845e-04	-1.597078e-03
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

In order to deploy the application, it must first be compiled on the target machine. To do this, you must use the command line option 'make' from same directory where the source code is located. So, the first thing is to position ourselves in the GPU-MILOS. 
There are two environment variables that you must define in your console, they are CUDA_PATH and SMS. The first one expresses the path where CUDA Toolkit is installed in your machine and the second one expresses the CUDA code compatibility with which it will be compiled (this value must be known from your graphic card specifications). Here is an example of how to create the two environment variables using a bash command console. If you want to make these two variables permanent, you can add them to your ~/.bashrc file.

```
export CUDA_PATH="/usr/local/cuda-10.1"
export SMS="35"
```

If these two environmental variables do not exist, by default the makefile compile the code with compatibility for this generations of CUDA: 35 37 50 52 60 70 , and using the CUDA_PATH  “/usr/local/cuda-10.1” . 

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

The program must be executed by passing the configuration file as a parameter. There is two examples in this repository. The first one is for execute a spectral synthesis:

```
./gmilos run/synthesis.mtrol
```

And the second makes an inversion over this synthesis

```
./gmilos run/invert.mtrol
```
In both cases, the results are stored in the directory **data**. 
