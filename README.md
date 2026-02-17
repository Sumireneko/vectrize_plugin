# Vectrize plugin for Krita  
ver 0.6  
This is an experimental plugin works on Krita,  
It can that tracing as  svg path on selected area or full image.  
Not need Node.js and Runtime etc.  

At Lasso draw mode(v0.35 later),Create a shape with freeform from Selection area.  
It can change the stroke and fill colors more easily than the Krita default method.  
(Fillcolor as Forground Color and Bordercolor as Background color )  
  
Python Porting from Imagetracer.js 1.2.6  
https://github.com/jankovicsandras/imagetracerjs  

# Version compatibility
v0.6  — Krita v5.2.14,v5.3.0 and v6.0.0 Later  
v0.5  — Krita v5.2.14 later  
v0.48 — Krita v5.2.2 later  
https://krita.org/en/

# How to install

**Important!**    
This plug-in need addict v2.4.0 library(Python)  
https://github.com/mewwts/addict  
Please Install this into pykrita directry too
Thanks to each libraries and application authors!  
```
pykrita/
    addict/ <----Install manually by copying the addict library files
    vectrize/<----Install via Krita’s built‑in plugin installer (Select the vectrize's zip file)
      :
    (other plugin directories)
      :
```


# Features
* Trace as filled color area and Line Art 
* Tracing Whole image or Selected area(Rectangle)
* Recommend image size: smaller than 756ｘ756 pixels  
* Lasso draw mode(use FreeHand Selection),(Fillcolor as Forground Color and Bordercolor as Background color )

# Update History
v0.6 - 2026/02/17
* Krita 5.3 and 6 compatible (PyQt5 / PyQt6 supported)  
* Added a compatibility layer for PyQt5 and PyQt6 (qt_compat.py).  
* This layer handles various API changes, including the transition from exec_() to exec(),   
* updated enum types, event handling differences, and changes in the UI file loader.  

v0.5 - 2025/10/19  
* Tested with Krita v5.2.14(PyQt5 and Python 3.13)  
* Preliminary PyQt6 compatibility added Updated import logic to support PyQt6 for future Krita 6.x compatibility.  
* Note: PyQt6 functionality has not been tested yet. This change is preparatory and not guaranteed to be stable.  
* Add "Keep active Layer" checkbox.  
* Revamped the button design, "Lasso mode" toggle button with grahical-icon .(When the button active,Automatically selects the "Freehand Selection Tool")  
* Improve user experience in Lasso Mode by allowing continuous vector object creation while keeping the paint layer active.  

v0.48 - 2024/04/02  
* Improve GUI layout,Add support tools  
* Add short cut button of Shape ordering and Shape style copy/paste commands  

v0.45 - 2024/03/31  
* Change GUI Layout  
* Add ToolTips(It appear by hover on a label)   
* Add Support Tool: Rid of Sawtooth ,Color Picker(Screen Sample), Shape 2 Selection  
* These are useful commands for edit Vector shapes on Krita

v0.40 - 2024/03/29  
* Update Manual.html (Tips to reduce for sawtooth outline  from selection)  
* Add ignore white pixel option
* Add alpha fillter
* Add pre-Alpha filter for Lasso draw mode  
  
v0.35 - 2024/03/27  
* Add Lasso draw mode  
* Fillcolor as Forground Color and Bordercolor as Background color )  
  
v0.31 - 2024/03/26   
* Improvements and bug fixes  
* If quantization flag to set 0,change to 1(default).  
* Updated Manual.html 
* (Regarding the relationship between image size and appropriate error threshold.)   
  
v0.3 - 2024/03/23  
* Improved to process speed x2

v0.25 - 2024/03/19  
* Fixed the process of converting from tuple to list.  
* Data loading speed is approximately 13 times faster.  
  
v0.15  
* initial release  

