# Vectrize plugin for Krita  
ver 0.31  
This is an experimental plugin works on Krita,  
It can that tracing as  svg path on selected area or full image.  
Not need Node.js and Runtime etc.  

Python Porting from Imagetracer.js 1.2.6  
https://github.com/jankovicsandras/imagetracerjs  

# How to install
To use plugin,Please Install this to  Krita v5.2.2 later  
https://krita.org/en/

Important!    
This plug-in need addict v2.4.0 library  
https://github.com/mewwts/addict  
Please Install this into pykrita directry too

Thanks to each libraries and application authors!  

# Features
* Trace as filled color area and Line Art 
* Tracing Whole image or Rectangle selection area
* Recommend image size: smaller than 756ｘ756 pixels  
  

# Update History
v0.31 - 2024/03/26   
Improvements and bug fixes  
If quantization flag to set 0,change to 1(default).  
Updated Manual.html 
(Regarding the relationship between image size and appropriate error threshold.)   
  
v0.3 - 2024/03/23  
Improved to process speed x2

v0.25 - 2024/03/19  
Fixed the process of converting from tuple to list.  
Data loading speed is approximately 13 times faster.  
  
v0.15  
initial release  

