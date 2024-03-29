# Vectrize plugin for Krita  
ver 0.40 
This is an experimental plugin works on Krita,  
It can that tracing as  svg path on selected area or full image.  
Not need Node.js and Runtime etc.  

At Lasso draw mode(v0.35 later),Draw shapes with free outlines and fills from Selection area.  
It can change the stroke and fill colors more easily than the Krita default method.  

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
* Tracing Whole image or Selected area(Rectangle)
* Recommend image size: smaller than 756ｘ756 pixels  
* Lasso draw mode,(Fillcolor as Forground Color and Bordercolor as Background color )
 

# Update History
v0.40 - 2024/03/29  
Update Manual.html (Tips to reduce for sawtooth outline  from selection)  
Add ignore white pixel option
Add alpha fillter
Add pre-Alpha filter for Lasso draw mode  
  
v0.35 - 2024/03/27  
Add Lasso draw mode  
Fillcolor as Forground Color and Bordercolor as Background color )  
  
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

