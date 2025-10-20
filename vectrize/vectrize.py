# ===============================================
# Vectrize plugin v0.5 for Krita 5.2.14 later
# ===============================================
# Copyright (C) 2025 L.Sumireneko.M
# This program is free software: you can redistribute it and/or modify it under the 
# terms of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with this program.
# If not, see <https://www.gnu.org/licenses/>. 


import math, random, os, copy, time, sys, array
from math import floor, ceil, sqrt
from random import randint
from io import BytesIO
from functools import lru_cache
from addict import Dict

import krita
try:
    if int(krita.qVersion().split('.')[0]) == 5:
        raise

    # PyQt6
    from PyQt6.QtWidgets import *
    from PyQt6.QtGui import *
    from PyQt6.QtCore import (
        QObject, QEvent, QTimer, QSignalBlocker, pyqtSignal, QPointF, Qt
    )

except:
    # PyQt5 fallback
    from PyQt5.QtWidgets import *
    from PyQt5.QtGui import *
    from PyQt5.QtCore import (
        QObject, QEvent, QTimer, QSignalBlocker, pyqtSignal, QPointF, Qt
    )
from krita import *

plugin_ver = "Vectrize v0.50"

# This plugin based on imagetracer.js v1.2.6  
# https://github.com/jankovicsandras/imagetracerjs

class ImageToSVGConverter():

    def __init__(self):
        self.versionnumber = '1.2.6'
        self.optionpresets = Dict({
            'default':{
                'corsenabled': False,
                'ltres': 0.5,
                'qtres': 4.0,
                'pathomit':6,
                'rightangleenhance': True,
                'colorsampling': 2,
                'numberofcolors':16,
                'mincolorratio': 0,
                'colorquantcycles':1,
                'layering': 0,
                'strokewidth': 0,
                'linefilter': False,
                'lineart': False,
                'scale': 1,
                'roundcoords':3,
                'viewbox': True,
                'desc': False,
                'lcpr': 0,
                'qcpr': 0,
                'blurradius': 0,
                'blurdelta': 20
                },
            'default126':{
                'corsenabled': False,
                'ltres': 1,
                'qtres': 1,
                'pathomit': 8,
                'rightangleenhance': True,
                'colorsampling': 5,
                'numberofcolors': 16,
                'mincolorratio': 0,
                'colorquantcycles': 3,
                'layering': 0,
                'strokewidth': 1,
                'linefilter': False,
                'scale': 1,
                'roundcoords': 1,
                'viewbox': False,
                'desc': False,
                'lcpr': 0,
                'qcpr': 0,
                'blurradius': 0,
                'blurdelta': 20
                },
            'posterized1':{
                'colorsampling': 0,
                'numberofcolors': 2
                },
            'posterized2':{
                'numberofcolors': 4,
                'blurradius': 5
                },
            'curvy':{
                'ltres': 0.01,
                'linefilter': True,
                'rightangleenhance': False
                },
            'sharp':{
                'qtres': 0.01,
                'linefilter': False
                },
            'detailed':{
                'pathomit': 0,
                'roundcoords': 2,
                'ltres': 0.5,
                'qtres': 0.5,
                'numberofcolors': 64
                },
            'smoothed':{
                'blurradius': 5,
                'blurdelta': 64
                },
            'grayscale':{
                'colorsampling': 0,
                'colorquantcycles': 1,
                'numberofcolors': 7
                },
            'fixedpalette':{
                'colorsampling': 0,
                'colorquantcycles': 1,
                'numberofcolors': 27
                },
            'randomsampling1':{
                'colorsampling': 1,
                'numberofcolors': 8
                },
            'randomsampling2':{
                'colorsampling': 1,
                'numberofcolors': 64
                },
            'artistic1':{
                'colorsampling': 0,
                'colorquantcycles': 1,
                'pathomit': 0,
                'blurradius': 5,
                'blurdelta': 64,
                'ltres': 0.01,
                'linefilter': True,
                'numberofcolors': 16,
                'strokewidth': 2
                },
            'artistic2':{
                'qtres': 0.01,
                'colorsampling': 0,
                'colorquantcycles': 1,
                'numberofcolors': 4,
                'strokewidth': 0
                },
            'artistic3':{
                'qtres': 10,
                'ltres': 10,
                'numberofcolors': 8
                },
            'artistic4':{
                'qtres': 10,
                'ltres': 10,
                'numberofcolors': 64,
                'blurradius': 5,
                'blurdelta': 256,
                'strokewidth': 2
                },
            'posterized3':{
                'ltres': 1,
                'qtres': 1,
                'pathomit': 20,
                'rightangleenhance': True,
                'colorsampling': 0,
                'numberofcolors': 3,
                'mincolorratio': 0,
                'colorquantcycles': 3,
                'blurradius': 3,
                'blurdelta': 20,
                'strokewidth': 0,
                'linefilter': False,
                'roundcoords': 1,
                'pal': Dict(
                    {
                    'r': 0,
                    'g': 0,
                    'b': 100,
                    'a': 255
                    },
                    {
                    'r': 255,
                    'g': 255,
                    'b': 255,
                    'a': 255
                    }
                    )
                    # palette end
                }
        })
        # end of option preset

        # Lookup tables for pathscan
        # pathscan_combined_lookup[ arr[py][px] ][ dir ] = [nextarrpypx, nextdir, deltapx, deltapy];

        self.pathscan_combined_lookup = [
            [[-1,-1,-1,-1], [-1,-1,-1,-1], [-1,-1,-1,-1], [-1,-1,-1,-1]],#arr[py][px]===0 is invalid
            [[ 0, 1, 0,-1], [-1,-1,-1,-1], [-1,-1,-1,-1], [ 0, 2,-1, 0]],
            [[-1,-1,-1,-1], [-1,-1,-1,-1], [ 0, 1, 0,-1], [ 0, 0, 1, 0]],
            [[ 0, 0, 1, 0], [-1,-1,-1,-1], [ 0, 2,-1, 0], [-1,-1,-1,-1]],
            
            [[-1,-1,-1,-1], [ 0, 0, 1, 0], [ 0, 3, 0, 1], [-1,-1,-1,-1]],
            [[13, 3, 0, 1], [13, 2,-1, 0], [ 7, 1, 0,-1], [ 7, 0, 1, 0]],
            [[-1,-1,-1,-1], [ 0, 1, 0,-1], [-1,-1,-1,-1], [ 0, 3, 0, 1]],
            [[ 0, 3, 0, 1], [ 0, 2,-1, 0], [-1,-1,-1,-1], [-1,-1,-1,-1]],
            
            [[ 0, 3, 0, 1], [ 0, 2,-1, 0], [-1,-1,-1,-1], [-1,-1,-1,-1]],
            [[-1,-1,-1,-1], [ 0, 1, 0,-1], [-1,-1,-1,-1], [ 0, 3, 0, 1]],
            [[11, 1, 0,-1], [14, 0, 1, 0], [14, 3, 0, 1], [11, 2,-1, 0]],
            [[-1,-1,-1,-1], [ 0, 0, 1, 0], [ 0, 3, 0, 1], [-1,-1,-1,-1]],
            
            [[ 0, 0, 1, 0], [-1,-1,-1,-1], [ 0, 2,-1, 0], [-1,-1,-1,-1]],
            [[-1,-1,-1,-1], [-1,-1,-1,-1], [ 0, 1, 0,-1], [ 0, 0, 1, 0]],
            [[ 0, 1, 0,-1], [-1,-1,-1,-1], [-1,-1,-1,-1], [ 0, 2,-1, 0]],
            [[-1,-1,-1,-1], [-1,-1,-1,-1], [-1,-1,-1,-1], [-1,-1,-1,-1]] #arr[py][px]===15 is invalid
        ]

        # Gaussian kernel table
        self.gks = [ 
            [0.27901,0.44198,0.27901],
            [0.135336,0.228569,0.272192,0.228569,0.135336],
            [0.086776,0.136394,0.178908,0.195843,0.178908,0.136394,0.086776],
            [0.063327,0.093095,0.122589,0.144599,0.152781,0.144599,0.122589,0.093095,0.063327],
            [0.049692,0.069304,0.089767,0.107988,0.120651,0.125194,0.120651,0.107988,0.089767,0.069304,0.049692]
        ]
        
        
        # workaround for fitseq() chain
        self.split_point_ex = {'is':False,'p':Dict(),'l':0,'q':0,'sp':0,'en':0}
        self.sep = os.sep
        self.debug_all_layer=''
        self.free_form_selection = False
        self.fgc_rgb = None
        self.bgc_rgb = None
        self.gray_mode = False
        self.warning_skip = False
        self.ignore_white = False
        self.lasso_draw_mode = False
        self.open_path_mode = False
        self.erode_selection = False
        self.select_mode={'is':False,'x':0,'y':0}
        self.continuous_draw = False

    def krita_load_node(self):
        krita_instance = Krita.instance()
        currentDoc = krita_instance.activeDocument()
        currentView = krita_instance.activeWindow().activeView()
        
        target_krita_node = currentDoc.activeNode()
        target_krita_node.setOpacity(255) 
        rgba = []
        
        if self.lasso_draw_mode:
            fg = self.fgcolor = currentView.foregroundColor()#.toQString()
            bg = self.bgcolor = currentView.backgroundColor()#.toQString()
            fgc = fg.components() # fgc[n] n = 0,1,2,3 : b g r a ,-> 2 1 0 3 = r,g,b,a
            bgc = bg.components()
            fgcd = self.fgc_rgb = Dict({'r' : floor(fgc[2] * 255.0),'g' : floor(fgc[1] * 255.0),'b' : floor(fgc[0] * 255.0),'a' : floor(fgc[3] * 255.0) })
            bgcd = self.bgc_rgb = Dict({'r' : floor(bgc[2] * 255.0),'g' : floor(bgc[1] * 255.0),'b' : floor(bgc[0] * 255.0),'a' : floor(bgc[3] * 255.0) })
            #message(f'''Foreground Color : R{fgcd['r']} G {fgcd['g']} B{fgcd['b']} A{fgcd['a']} ''')
            #message(f'''Background Color : R{bgcd['r']} G {bgcd['g']} B{bgcd['b']} A{bgcd['a']} ''')
        
        # Image size and Selection area
        x=y=0;w = currentDoc.width()+1;h = currentDoc.height()+1;
        qbin=None;mbin=mimg=img=None;
        dx,dy,dw,dh = 0,0,1,1
        
        self.select_mode['is'] = False
        self.free_form_selection = False
        # print("Image Size:",x,y,w,h)
        
        # https://scripting.krita.org/lessons/image-data
        # https://krita-artists.org/t/pixel-perfect-line-setting-for-pixel-art-brushes/42629/5
        # https://krita-artists.org/t/python-how-to-apply-blur-to-selected-part-of-layer-only/48884/
        if target_krita_node.colorModel() == 'RGBA'  and target_krita_node.type() == 'paintlayer':
            
            if currentDoc.selection() is not None: 
                # If selection exist
                b = currentDoc.selection()
                bx, by, bw, bh = b.x(), b.y(), b.width(), b.height()
                
                # Clopped by canvas
                if bx < 0:bx = 0
                if by < 0:by = 0
                if bx + bw >= w:bw = w-bx
                if by + bh >= h:bh = h-by
                # print("Selection:",bx,by,bw,bh)
                
                # Get selection maskdata (B/W pixel data)
                mimg = b.pixelData(bx,by,bw,bh)# x00 per 1px for Feather
                
                # Detect which the selection is freeform or rectangle.
                if b'\x00' in mimg:
                    # Freeform selection by lasso or wand tool
                    # message('x00 detect!')
                    self.free_form_selection = True
                    # swap black and white pixel
                    mimg = mimg.replace(b'\x00',b'\xee').replace(b'\xff',b'\x00').replace(b'\xee',b'\xff')
                    # Convert from B/W (8bit x1) to RGBA8888 (8bit x 4)
                    e = bytearray()
                    for s in mimg:
                        if s != b'\xff':e = e + bytearray(s + s + s + b'\xff') # not white (Alpha:255)
                        else:e = e + bytearray(s + s + s + b'\x00') # white (Alpha:0)
                    mimg = e
                    mbin = QImage(mimg, bw, bh, QImage.Format_RGBA8888)
                    mbin = mbin.transformed(QTransform().rotate(-90),Qt.SmoothTransformation)# rotate 90 for avoid bug


                else:
                    # Rectangle selection
                    img = target_krita_node.projectionPixelData(bx,by,bw,bh)#bytearray()
                    qbin = QImage(img, bw, bh, QImage.Format_RGBA8888).rgbSwapped()
                dx,dy,dw,dh = bx,by,bw,bh;
                # offset 
                self.select_mode['is'] = True
                self.select_mode['x'] = floor(bx*0.5)
                self.select_mode['y'] = floor(by*0.5)
            else:
                # full layer
                img = target_krita_node.pixelData(x,y,w,h)#bytearray()
                # print(img) #b\xff0...
                qbin = QImage(img, w, h, QImage.Format_RGBA8888).rgbSwapped()# rotate 90 for avoid bug
                dx,dy,dw,dh = 0,0,w,h;

            # Convert to binary -> [r,g,b,a]
            rgba_extend = rgba.extend
            
            if self.free_form_selection:
                mbin_pixelColor = mbin.pixelColor
                for xx in range(0,dh):
                    for yy in range(dw-1,0,-1):
                        if xx == dh:continue # for avoid bug
                        if yy == dw:continue # for avoid bug
                        m = mbin_pixelColor(xx,yy)
                        r = floor(m.red())
                        g = floor(m.green())
                        b = floor(m.blue())
                        a = floor(m.alpha())
                        if a > 225:r=g=b=0;a=255 # Remove annoying Unnecessary path,Helpfull!
                        rgba_extend([r,g,b,a])
            else:
                qbin = qbin.transformed(QTransform().rotate(-90))
                qbin_pixelColor = qbin.pixelColor
                for xx in range(0,dh):
                    for yy in range(dw-1,0,-1):
                        if xx == dh:continue
                        if yy == dw:continue
                        pixc = qbin_pixelColor(xx, yy)
                        r = floor(pixc.red())
                        g = floor(pixc.green())
                        b = floor(pixc.blue())
                        a = 255 # floor(pixc.alpha())
                        #print(r,g,b,a)
                        rgba_extend([r,g,b,a])
        #rgba = array.array('i',rgba);
        return rgba,dw-1,dh-1



    def krita_save_node(self,svgdata):
        currentDoc = Krita.instance().activeDocument()
        working_paint_layer = currentDoc.activeNode()
        print("Saving current layer:", working_paint_layer.name(), working_paint_layer.type())

        root = currentDoc.rootNode()

        if len(svgdata) > 0:
            vnode = currentDoc.createVectorLayer('vectorized data')
            vnode.addShapesFromSvg(svgdata)
            root.addChildNode(vnode , None )

        if self.continuous_draw==True:
            QTimer.singleShot(0, lambda: currentDoc.setActiveNode(working_paint_layer))

        return

    def kritaToSVG(self):
       set_option = 'krita_set'
       # load the noad
       bin_data,w,h = self.krita_load_node()
       
       # set image data
       imgd = Dict()
       imgd['width'],imgd['height'] = w,h
       imgd['data'] = bin_data
       
       # set options and make svg data
       options = self.checkoptions(set_option)
       if (options['lineart'] == True) and (options['strokewidth'] <= 0): options['strokewidth'] = 2
       td = self.imagedataToTracedata(imgd, options )
       svgdata = self.getsvgstring(td, options,True)
       
       # for full image
       self.krita_save_node(svgdata)
       
       self.select_mode['is'] = False
       print('Done!')


    def imagedataToTracedata(self,imgd,options):
        # 1. Color quantization
        # ii = -1,1,0 sheet map
        ii = self.colorquantization(imgd, options)
        # mode = ['Sequential layering mode(Layering:0)', 'Pallalel layering mode(Layering:1)']
        # print(mode[options['layering']])
        if options.layering == 0:
            # Sequential layering
            # create tracedata object
            ii_array = ii['array']
            ii_palette = ii['palette']
            tracedata = Dict({
                'layers' : [],
                'palette' : ii_palette,
                'width' : len(ii_array[0]) - 2,
                'height': len(ii_array) - 2
                })
            tracedata_layers_append = tracedata['layers'].append
            # Loop to trace each color layer
            # print('Points data crete: ',time.time() - st)
            colornum = 0
            for colornum in range(0,len(ii_palette)):
                # layeringstep -> pathscan -> internodes -> batchtracepaths
                s_lay = self.layeringstep(ii, colornum)
                #print(s_lay)
                p_lay = self.pathscan(s_lay, options['pathomit'])
                #print(p_lay)
                i_lay = self.internodes(p_lay, options)
                #print(i_lay)
                tracedlayer = self.batchtracepaths(i_lay, options['ltres'], options['qtres'])
                #tracedlayer = self.batchtracepaths(self.internodes(self.pathscan(self.layeringstep(ii, colornum), options.pathomit), options), options.ltres, options.qtres)
                # adding traced layer
                #print(tracedlayer)
                tracedata_layers_append(tracedlayer)
            # End of color loop
        else:
            # Parallel layering
            # 2. Layer separation and edge detection
            ls = self.layering(ii)
            # Optional edge node visualization
            if options.layercontainerid:
                self.drawLayers(ls, self.specpalette, options['scale'], options['layercontainerid'])
            # 3. Batch pathscan(Dict)
            bps = self.batchpathscan(ls, options['pathomit'])
            # 4. Batch interpollation(Dict)
            bis = self.batchinternodes(bps, options)
            # 5. Batch tracing and creating tracedata object
            # batchtracelayers return []
            tracedata = Dict({
                'layers': self.batchtracelayers(bis,options['ltres'], options['qtres']),
                'palette': ii['palette'],
                'width': imgd['width'],
                'height': imgd['height']
                })
        # End of parallel layering
        return tracedata

    def checkoptions(self,options):
        options = options if options == None else Dict()
        # Option preset
        if (type(options) is str):
            options = options.lower()
            if self.optionpresets[options]:
                options = self.optionpresets[options]
            else:
                options = Dict()
            # Defaults
        # print(self.optionpresets['default'].keys())
        ok = self.optionpresets['default'].keys()#list()
        # print(str(type(options)))
        for k in ok:
            # print(f'{k}')
            if options.get(k, None) == None:
                options[k] = self.optionpresets['default'][k]
        return options

    #////////////////////////////////////////////////////////////
    #//
    #//  Vectorizing functions
    #//
    #////////////////////////////////////////////////////////////
    
    # 1. Color quantization
    # Using a form of k-means clustering repeatead options.colorquantcycles times. http://en.wikipedia.org/wiki/Color_quantization

    def colorquantization(self,imgd, options):
        arr = array.array('i',[]);#[]
        copy_deepcopy = copy.deepcopy
        
        idx = 0
        paletteacc = Dict()
        palette = Dict()
        
        idata = imgd['data']
        w = imgd['width']
        h = imgd['height']
        pixelnum = w * h
        
        num_of_colors = int(options['numberofcolors'])
        if num_of_colors < 2 : num_of_colors == 2
        if num_of_colors > 256 : num_of_colors == 255
        
        sampling = options['colorsampling']
        color_q_cycles = options['colorquantcycles']
        if color_q_cycles <= 0:color_q_cycles = 1
        cycle_limit = color_q_cycles - 1
        min_col_ratio = options['mincolorratio']
        
        # imgd.data must be RGBA, not just RGB
        if len(idata) < pixelnum * 4:
            alloc = pixelnum * 4
            newimgddata = [0] * alloc
            #newimgddata = np.array(([0] * alloc),dtype='uint8') # Uint8,  ClampedArray?
            for pxcnt in range(0,pixelnum):
                imgq = idata[pxcnt*3:pxcnt*3+3]
                p4 = pxcnt*4
                newimgddata[p4] = imgq[0]
                newimgddata[p4 + 1] = imgq[1]
                newimgddata[p4 + 2] = imgq[2]
                newimgddata[p4 + 3] = 255
            imgd['data'] = newimgddata
        # End of RGBA imgd.data check
        # Filling arr (color index array) with -1
        print(f'Image size = height:{imgd.height}px * width:{imgd.width}px')
        arr = [[-1] * (w+2) for _ in range(h+2)]
        #arr = np.full((h + 2,w + 2), -1)
        
        # Use custom palette if pal is defined or sample / generate custom length palette
        # parette is Dict() object
        if options['pal']:palette = options['pal']
        elif sampling == 0:palette = self.generatepalette(num_of_colors)
        elif sampling == 1:palette = self.samplepalette(num_of_colors, imgd)
        else:palette = self.samplepalette2(num_of_colors, imgd)
        # Selective Gaussian blur preprocessing
        if options['blurradius'] > 0:
            imgd = self.blur( imgd, options['blurradius'], options['blurdelta'] )
        # Repeat clustering step options.colorquantcycles times
        # print('createPalette: ',time.time() - st)
        
        plen = len(palette)
        idt_palette = Dict()
        idt_palette = [ Dict({'r': 0,'g': 0,'b': 0,'a': 0,'n': 0 }) for _ in range(0,plen) ]
        
        # print('Pixel Analysis: ',time.time() - st)

        idata = imgd['data']
        for cnt in range(0,color_q_cycles):
            # Average colors from the second iteration
            if cnt > 0:
                # This indent fixed
                # Reseting palette accumulator for averaging
                paletteacc = copy_deepcopy(idt_palette)
                # averaging paletteacc for palette
                for k in range(0, len(palette)):
                    # averaging
                    #print( paletteacc )
                    kpal = paletteacc[k]
                    n = kpal['n']
                    if n:palette[k] = self.ret_pal(kpal['r'],kpal['g'],kpal['b'],kpal['a'],n)

                    # Randomizing a color, if there are too few pixels and there will be a new cycle len(paletteacc[k])>0 and 
                    if (n / pixelnum < min_col_ratio) and (cnt < cycle_limit):
                        palette[k] =Dict({
                            'r': randint(0, 255),
                            'g': randint(0, 255),
                            'b': randint(0, 255),
                            'a': randint(0, 255)
                            })
            # End of palette loop
            # End of Average colors from the second iteration
            # Reseting palette accumulator for averaging
            #print('Quantize Cycle',cnt,"_",time.time() - st)
            
            paletteacc=copy_deepcopy(idt_palette)
            for ｊ in range(0,h):
                for i in range(0,w):
                    # pixel index
                    idx = (j * w + i) * 4
                    # find closest color from palette by measuring (rectilinear) color distance between this pixel and all palette colors
                    ci = 0;cdl = 1024;# 4 * 256 is the maximum RGBA distance 
                    q0,q1,q2,q3 = idata[idx:idx+4]# get a list of imgd.data[idx] to imgd.data[idx + 3]

                    for k in range(0,plen):
                        # In my experience, https://en.wikipedia.org/wiki/Rectilinear_distance works better than https://en.wikipedia.org/wiki/Euclidean_distance
                        kpal = palette[k]
                        cd = self.min_max(kpal['r'],kpal['g'],kpal['b'],kpal['a'],q0,q1,q2,q3) # bottle neck fixed (3.1sec)
                        """
                        c1 = kpal['r']- q0 if kpal['r'] > q0 else q0 - kpal['r']
                        c2 = kpal['g']- q1 if kpal['g'] > q1 else q1 - kpal['g']
                        c3 = kpal['b']- q2 if kpal['b'] > q2 else q2 - kpal['b']
                        c4 = kpal['a']- q3 if kpal['a'] > q3 else q3 - kpal['a']
                        cd = intc1 + c2 + c3 + c4
                        """
                        # Remember this color if this is the closest yet
                        if cd < cdl:cdl = cd;ci = k;
                    # End of palette loop
                    # add to palettacc
                    paletteacc[ci]['r'] = paletteacc[ci]['r'] + q0
                    paletteacc[ci]['g'] = paletteacc[ci]['g'] + q1
                    paletteacc[ci]['b'] = paletteacc[ci]['b'] + q2
                    paletteacc[ci]['a'] = paletteacc[ci]['a'] + q3
                    paletteacc[ci]['n'] = paletteacc[ci]['n'] + 1
                    # update the indexed color array
                    arr[j + 1][i + 1] = ci
                    # End of i loop
                # End of j loop
        # End of cnt loop

        return Dict({ 'array':arr, 'palette':palette })

    @lru_cache(maxsize=1024)
    def ret_pal(self,r,g,b,a,n):
        return Dict({
            'r': floor(r / n),
            'g': floor(g / n),
            'b': floor(b / n),
            'a': floor(a / n)
            })


    @lru_cache(maxsize=1024)
    def min_max(self,r,g,b,a,q0,q1,q2,q3):
        return int( (max([r, q0]) - min([r, q0]))+(max([g, q1]) - min([g, q1]))+(max([b, q2]) - min([b, q2]))+(max([a, q3]) - min([a, q3])) )



    # Sampling a palette from imagedata
    def samplepalette(self,numberofcolors,imgd):
        idx=0;palette = Dict();
        imgseed = len(imgd['data'])/4
        for i in range(0,numberofcolors):
            idx = randint(0, imgseed) * 4
            imgq = imgd['data'][idx:idx+4]# get a list of imgd.data[idx] to imgd.data[idx + 3]
            palette[i]=Dict({
                'r': imgq[0],
                'g': imgq[1],
                'b': imgq[2],
                'a': imgq[3]
            })
        return palette

    # Deterministic sampling a palette from imagedata: rectangular grid
    def samplepalette2(self,numberofcolors, imgd ):
        idx = 0;pdx = 0;
        palette = Dict()
        ni = ceil(sqrt(numberofcolors))
        nj = ceil(numberofcolors / ni)
        w = imgd['width']
        vx = w / (ni + 1)
        vy = imgd['height'] / (nj + 1)
        idata = imgd['data']
        l = len(palette)
        for j in range(0,nj):
            for i in range(0,nj):
                if l == numberofcolors:
                    break
                else:
                    idx = floor((j + 1) * vy * w + (i + 1) * vx) * 4
                    imgq = idata[idx:idx+4]# get a list of imgd.data[idx] to imgd.data[idx + 3]
                    palette[pdx]=Dict({
                        'r': imgq[0],
                        'g': imgq[1],
                        'b': imgq[2],
                        'a': imgq[3]
                        })
                    pdx = pdx + 1
        return palette



    #Generating a gray palette
    def generate_gray(self,numberofcolors):
        palette = Dict()
        rcnt=gcnt=bcnt=0;
        if numberofcolors < 255:
            # Grayscale
            step_div = numberofcolors - 1
            if step_div == 0:step_div = 1
            graystep = floor(255 / step_div)
            for i in range(0,numberofcolors):
                gr = i * graystep
                palette[i]=Dict({
                    'r': gr,
                    'g': gr,
                    'b': gr,
                    'a': 255
                    })
        # End of numberofcolors check
        return palette



    #Generating a palette with numberofcolors
    def generatepalette(self,numberofcolors):
        palette = Dict()
        rcnt=gcnt=bcnt=0;
        if numberofcolors < 8:
            # Grayscale
            step_div = numberofcolors - 1
            if step_div == 0:step_div = 1
            graystep = floor(255 / step_div)
            for i in range(0,numberofcolors):
                gr = i * graystep
                palette[i]=Dict({
                    'r': gr,
                    'g': gr,
                    'b': gr,
                    'a': 255
                    })
        else:
            # RGB color cube
            colorqnum = floor(numberofcolors ** (1 / 3))
            colorstep = floor(255 / (colorqnum - 1))
            rndnum = numberofcolors - colorqnum * colorqnum * colorqnum
            # number of random colors
            pdx = 0;
            for rcnt in range(0,colorqnum):
                rr = rcnt * colorstep
                for gcnt in range(0,colorqnum):
                    gg = gcnt * colorstep
                    for bcnt in range(0,colorqnum):
                        bb = bcnt * colorstep
                        palette[pdx]=Dict({
                            'r': rr,
                            'g': gg,
                            'b': bb,
                            'a': 255
                            })
                        pdx = pdx + 1
                    # End of blue loop
                # End of green loop
            # End of red loop
            # Rest is random
            for rcnt in range(0,rndnum):
                palette[pdx]=Dict({
                    'r': randint(0, 255),
                    'g': randint(0, 255),
                    'b': randint(0, 255),
                    'a': 255 #floor(randint(0, 255))
                })
                pdx = pdx + 1
        # End of numberofcolors check
        return palette

    # 2. Layer separation and edge detection
    # Edge node types ( ▓: this layer or 1; ░: not this layer or 0 )
    # 12  ░░  ▓░  ░▓  ▓▓  ░░  ▓░  ░▓  ▓▓  ░░  ▓░  ░▓  ▓▓  ░░  ▓░  ░▓  ▓▓
    # 48  ░░  ░░  ░░  ░░  ░▓  ░▓  ░▓  ░▓  ▓░  ▓░  ▓░  ▓░  ▓▓  ▓▓  ▓▓  ▓▓
    #     0   1   2   3   4   5   6   7   8   9   10  11  12  13  14  15

    def layering(self,ii):
        # Creating layers for each indexed color in arr
        layers = []
        val = 0
        ii_array = ii['array']
        ah = len(ii_array)
        aw = len(ii_array[0])
        ak = len(ii['palette'])
        # Create layers
        # 3 Dimension array
        layers = [[[0] * aw for _ in range(0, ah)]for _ in range(0, ak)]

        # Looping through all pixels and calculating edge node type
        for j in range(1,ah-1):
            for i in range(1,aw-1):
                # This pixel's indexed color
                val = ii_array[j][i]
                
                # Are neighbor pixel colors the same?
                n1 = 1 if ii_array[j - 1][i - 1] == val else 0
                n2 = 1 if ii_array[j - 1][i    ] == val else 0
                n3 = 1 if ii_array[j - 1][i + 1] == val else 0
                n4 = 1 if ii_array[j    ][i - 1] == val else 0
                n5 = 1 if ii_array[j    ][i + 1] == val else 0
                n6 = 1 if ii_array[j + 1][i - 1] == val else 0
                n7 = 1 if ii_array[j + 1][i    ] == val else 0
                n8 = 1 if ii_array[j + 1][i + 1] == val else 0
                
                # this pixel's type and looking back on previous pixels
                layers[val][j + 1][i + 1] = 1 + n5 * 2 + n8 * 4 + n7 * 8
                if not n4:layers[val][j + 1][i    ] = 0 + 2 + n7 * 4 + n6 * 8
                if not n2:layers[val][j    ][i + 1] = 0 + n3 * 2 + n5 * 4 + 8
                if not n1:layers[val][j    ][i    ] = 0 + n2 * 2 + 4 + n4 * 8
            # End of i loop
        # End of j loop
        return layers


    # 2. Layer separation and edge detection
    # Edge node types ( ▓: this layer or 1; ░: not this layer or 0 )
    # 12  ░░  ▓░  ░▓  ▓▓  ░░  ▓░  ░▓  ▓▓  ░░  ▓░  ░▓  ▓▓  ░░  ▓░  ░▓  ▓▓
    # 48  ░░  ░░  ░░  ░░  ░▓  ░▓  ░▓  ░▓  ▓░  ▓░  ▓░  ▓░  ▓▓  ▓▓  ▓▓  ▓▓
    #     0   1   2   3   4   5   6   7   8   9   10  11  12  13  14  15

    def layeringstep(self,ii, cnum):
        # Creating layers for each indexed color in arr
        layer = []
        ii_array = ii['array']
        ah = len(ii_array)
        aw = len(ii_array[0])

        # Create layer
        layer = [[0] * aw for _ in range(ah)]
        #layer = np.full((ah, aw), 0)

        # Looping through all pixels and calculating edge node type
        for j in range(1,ah):
            for i in range(1,aw):
                c0 = 1 if ii_array[j - 1][i - 1] == cnum else 0
                c1 = 2 if ii_array[j - 1][i] == cnum else 0
                c2 = 8 if ii_array[j][i - 1] == cnum else 0
                c3 = 4 if ii_array[j][i] == cnum else 0
                layer[j][i] = c0+c1+c2+c3
            # End of i loop
        # End of j loop
        # return single layer
        return layer
    
    # Point in polygon test
    def pointinpoly(self,p, pa):
        isin = False
        max_pa = len(pa)
        for i in range(0,max_pa):
            if i==0:
                j = max_pa - 1
            else:
                j = i - 1
            pajx = pa[j]['x'];pajy = pa[j]['y'];
            paix = pa[i]['x'];paiy = pa[i]['y'];
            px = p['x'];py = p['y']
            # zero divide check
            m1 = (pajx - paix) * (py - paiy)
            m2 = (pajy - paiy)
            
            if m1 == 0:m1 = 0.1#print('zerodiv', f'{pajx} - {paix} * {p.y} - {paiy}')
            if m2 == 0:m2 = 0.1#print('zerodiv2', f'{pajy} - {paiy}') 
            #if pa[i].y > p.y != pa[j].y > p.y and p.x < (pa[j].x - (pa[i].x)) * (p.y - (pa[i].y)) / (pa[j].y - (pa[i].y)) + pa[i].x:
            if ((paiy > py) != (pajy > py)) and (px < (m1 / m2 + paix)):
                isin = not isin
            else:
                isin = isin
        return isin

    # 3. Walking through an edge node array, discarding edge node types 0 and 15 and creating paths from the rest.
    # Walk directions (dir): 0 > ; 1 ^ ; 2 < ; 3 v 

    def pathscan(self,arr, pathomit ):
        
        paths = Dict()
        paths_popitem = paths.popitem
        dir=pacnt=pcnt=px=py=0;
        w = len(arr[0]);h = len(arr);
        pathfinished = True;holepath = False;lookuprow = None;
        for j in range(0,h):
            for i in range(0,w):
                # This is original arr[py(j)][px(i)]
                arr_in = arr[j][i]
                if (arr_in == 4) or (arr_in == 11):# Other values are not valid
                    # Init
                    py = j;px = i;
                    paths[pacnt] = Dict({'points' : Dict(), 'boundingbox' : [px,py,px,py],'holechildren':[] })
                    pathfinished = False
                    pcnt = 0
                    holepath = True if arr_in == 11 else False
                    dir = 1
                    # Path points loop
                    while not pathfinished:
                        # New path point
                        pxbk = px - 1;
                        pybk = py - 1;
                        paths[pacnt]['points'][pcnt] = Dict({ 'x' : pxbk, 'y' : pybk, 't': arr[py][px] })
                        
                        # Bounding box
                        bb = paths[pacnt]['boundingbox']
                        if pxbk < bb[0]: paths[pacnt]['boundingbox'][0] = pxbk
                        if pxbk > bb[2]: paths[pacnt]['boundingbox'][2] = pxbk
                        if pybk < bb[1]: paths[pacnt]['boundingbox'][1] = pybk
                        if pybk > bb[3]: paths[pacnt]['boundingbox'][3] = pybk
                        
                        # Next: look up the replacement, direction and coordinate changes = clear this cell, turn if required, walk forward
                        lookuprow = self.pathscan_combined_lookup[arr[py][px]][dir]
                        # This is updated arr[py(j)][px(i)]
                        arr[py][px] = lookuprow[0]
                        dir = lookuprow[1]
                        px = px + lookuprow[2]
                        py = py + lookuprow[3]
                        
                        # Close path
                        if ((px - 1) == paths[pacnt]['points'][0]['x']) and ((py - 1) == paths[pacnt]['points'][0]['y']):
                            pathfinished = True
                            # Discarding paths shorter than pathomit
                            if len(paths[pacnt]['points']) < pathomit:
                                paths_popitem()
                            else:
                                if holepath==True:paths[pacnt]['isholepath'] = True
                                else:paths[pacnt]['isholepath'] = False
                                
                                # Finding the parent shape for this hole
                                if holepath == True:
                                    parentidx = 0
                                    parentbbox = [-1,-1,w+1,h+1]
                                    pts0 = paths[pacnt].points[0] # Dict()
                                    pc = paths[pacnt]['boundingbox'] # list()
                                    for parentcnt in range(0,pacnt):
                                        prc = paths[parentcnt]['boundingbox'] # list()
                                        pts = paths[parentcnt]['points'] # Dict()
                                        if (not paths[parentcnt]['isholepath']) and self.boundingboxincludes(prc, pc) and self.boundingboxincludes(parentbbox, prc) and self.pointinpoly(pts0, pts):
                                            parentidx = parentcnt
                                            parentbbox = prc
                                    paths[parentidx]['holechildren'].append(pacnt)
                                # End of holepath parent finding
                                pacnt = pacnt + 1
                        # End of Close path
                        pcnt = pcnt + 1
                    # End of Path points loop
                # End of Follow path
            # End of i loop
        # End of j loop
        return paths

    def boundingboxincludes(self,parentbbox, childbbox ):
        if len(parentbbox) == 0:return False
        if len(childbbox) == 0:return False
        #print(parentbbox,childbbox)
        return ( ( parentbbox[0] < childbbox[0] ) and ( parentbbox[1] < childbbox[1] ) and ( parentbbox[2] > childbbox[2] ) and ( parentbbox[3] > childbbox[3] ) )

    # 3. Batch pathscan
    def batchpathscan(self,layers, pathomit):
        bpaths=Dict()
        for k in range(0,len(layers)):
            bpaths[k]=self.pathscan(layers[k], pathomit)
        return bpaths


    # 4. interpollating between path points for nodes with 8 directions ( East, SouthEast, S, SW, W, NW, N, NE )
    def internodes(self,paths, options):
        ins = Dict()
        palen=nextidx=nextidx2=previdx=previdx2=0;
        opt_r_angle_ehnc = options['rightangleenhance']
        
        # Select mode offset
        ofx=ofy=0;
        if self.select_mode['is'] == True:ofx,ofy = self.select_mode['x'],self.select_mode['y'];
        # paths loop
        for pacnt in range(0,len(paths)):
            path_pacnt = paths[pacnt]
            ins[pacnt] = Dict({
                'points': [] ,
                'boundingbox' : path_pacnt['boundingbox'],
                'holechildren' : path_pacnt['holechildren'],
                'isholepath' : path_pacnt['isholepath']
                })
            ins_pacnt_append = ins[pacnt]['points'].append
            paths_pacnt_points = path_pacnt['points']
            palen = len(paths_pacnt_points)
            # pathpoints loop
            for pcnt in range(0,palen):
                # next and previous point indexes
                nextidx = (pcnt + 1) % palen;
                nextidx2 = (pcnt + 2) % palen;
                previdx = (pcnt - 1 + palen) % palen;
                previdx2 = (pcnt - 2 + palen) % palen;
                
                pppsx = paths_pacnt_points[pcnt]['x']
                pppsy = paths_pacnt_points[pcnt]['y']
                ppps_next_x = paths_pacnt_points[nextidx]['x']
                ppps_next_y = paths_pacnt_points[nextidx]['y']
                
                # right angle enhance
                if opt_r_angle_ehnc and self.testrightangle(paths_pacnt_points, previdx2, previdx, pcnt, nextidx, nextidx2):
                    # Fix previous direction
                    ipcp_last = len(ins[pacnt]['points']) - 1
                    if ipcp_last > 0:
                        ins[pacnt]['points'][ipcp_last]['linesegment'] = self.getdirection(ins[pacnt]['points'][ipcp_last]['x'], ins[pacnt]['points'][ipcp_last]['y'], pppsx, pppsy)
                    # This corner point
                    ins_pacnt_append(Dict({
                        'x': pppsx + ofx,
                        'y': pppsy + ofy,
                        'linesegment': self.getdirection(pppsx, pppsy, (pppsx + ppps_next_x) / 2, (pppsy + ppps_next_y) / 2)
                        }))
                # End of right angle enhance
                # interpolate between two path points
                
                ins_pacnt_append(Dict({
                    'x': (pppsx + ppps_next_x) / 2 + ofx,
                    'y': (pppsy + ppps_next_y) / 2 + ofy,
                    'linesegment': self.getdirection((pppsx + ppps_next_x) / 2, (pppsy + ppps_next_y) / 2, (ppps_next_x + paths_pacnt_points[nextidx2]['x']) / 2, (ppps_next_y + paths_pacnt_points[nextidx2]['y']) / 2)
                    }))
            # End of pathpoints loop
        # End of paths loop
        return ins

    def testrightangle(self,ppts, idx1, idx2, idx3, idx4, idx5):
        return ((ppts[idx3]['x'] == ppts[idx1]['x']) and (ppts[idx3]['x'] == ppts[idx2]['x']) and (ppts[idx3]['y'] == ppts[idx4]['y']) and (ppts[idx3]['y'] == ppts[idx5]['y'])) or ((ppts[idx3]['y'] == ppts[idx1]['y']) and (ppts[idx3]['y'] == ppts[idx2]['y']) and (ppts[idx3]['x'] == ppts[idx4]['x']) and (ppts[idx3]['x'] == ppts[idx5]['x']))

    @lru_cache(maxsize=1000)
    def getdirection(self, x1, y1, x2, y2):
        val = 8
        if x1 < x2:
            if y1 < y2: val = 1 # SE
            elif y1 > y2:val = 7 # NE
            else:val = 0 # E
        elif x1 > x2:
            if y1 < y2:val = 3 # SW
            elif y1 > y2:val = 5 # NW
            else:val = 4 # W
        else:
            if y1 < y2:val = 2 # S
            elif y1 > y2:val = 6 # N
            else:val = 8 # center, this should not happen
        return val

    # 4. Batch interpollation
    
    def batchinternodes(self,bpaths, options):
        binternodes = Dict()
        for k in bpaths:
            binternodes[k]=self.internodes(bpaths[k], options)
        return binternodes

    # 5. tracepath() : recursively trying to fit straight and quadratic spline segments on the 8 direction internode path
    
    # 5.1. Find sequences of points with only 2 segment types
    # 5.2. Fit a straight line on the sequence
    # 5.3. If the straight line fails (distance error > ltres), find the point with the biggest error
    # 5.4. Fit a quadratic spline through errorpoint (project this to get controlpoint), then measure errors on every point in the sequence
    # 5.5. If the spline fails (distance error > qtres), find the point with the biggest error, set splitpoint = fitting point
    # 5.6. Split sequence and recursively apply 5.2. - 5.6. to startpoint-splitpoint and splitpoint-endpoint sequences

    def tracepath(self,path, ltres, qtres):
        pcnt = seqend = 0;segtype1=segtype2='';smp = Dict();
        smp['segments'] = []
        smp['boundingbox'] = path['boundingbox']
        smp['holechildren'] = path['holechildren']
        smp['isholepath'] = bool(path['isholepath'])
        smp_segments_append = smp['segments'].append
        
        path_pts = path['points']
        pcnt_max = len(path_pts)
        while pcnt < pcnt_max:
            # 5.1. Find sequences of points with only 2 segment types
            segtype1 = path_pts[pcnt]['linesegment']
            segtype2 = -1
            seqend = pcnt + 1
            
            while ((path_pts[seqend]['linesegment'] == segtype1) or (path_pts[seqend]['linesegment'] == segtype2) or (segtype2 == -1)) and (seqend < pcnt_max - 1):
                if (path_pts[seqend]['linesegment'] != segtype1) and (segtype2 == -1):
                    segtype2 = path_pts[seqend]['linesegment']
                seqend = seqend + 1
            if seqend == pcnt_max - 1:seqend = 0
            # 5.2. - 5.6. Split sequence and recursively apply 5.2. - 5.6. to startpoint-splitpoint and splitpoint-endpoint sequences
            # smp.segments = smp.segments.extend(self.fitseq(path, ltres, qtres, pcnt, seqend))
            
            # fitseq chain
            self.split_point_ex['is'] = False
            smp_segments_append(self.fitseq(path, ltres, qtres, pcnt, seqend))
            
            if self.split_point_ex['is'] == True:
                self.split_point_ex['is'] = False
                spx = self.split_point_ex
                smp_segments_append(self.fitseq(spx['p'],spx['l'],spx['q'],spx['sp'],spx['en']))
                # stupid failsafe
                if self.split_point_ex['is'] == True:
                    self.split_point_ex['is'] = False
                    spx = self.split_point_ex
                    smp_segments_append(self.fitseq(spx['p'],spx['l'],spx['q'],spx['sp'],spx['en']))
                    if self.split_point_ex['is'] == True:
                        self.split_point_ex['is'] = False
                        spx = self.split_point_ex
                        smp_segments_append(self.fitseq(spx['p'],spx['l'],spx['q'],spx['sp'],spx['en']))
            # forward pcnt;
            if seqend > 0:
                pcnt = seqend
            else:
                pcnt = len(path_pts)
        # End of pcnt loop
        return smp



    # 5.2. - 5.6. recursively fitting a straight or quadratic line segment on this sequence of path nodes,
    # called from tracepath()

    def fitseq(self, path, ltres, qtres, seqstart, seqend ):
        # return if invalid seqend
        #print("Seq St", seqstart,"Seq En",seqend)

        path_points_len = len(path['points'])        
        if (seqend > path_points_len) or (seqend < 0):return Dict()
        
        # variables
        path_pts = path['points']

        px=py=errorval=ofx=ofy=0;
        # Select mode offset
        if self.select_mode['is'] == True:ofx,ofy = self.select_mode['x'],self.select_mode['y'];
        errorpoint = seqstart;
        curvepass = True
        dist2 = None;
        tl = seqend - seqstart
        if tl < 0:tl = tl + path_points_len
        
        st_x = path_pts[seqstart]['x'];st_y = path_pts[seqstart]['y'];
        en_x = path_pts[seqend]['x'];en_y = path_pts[seqend]['y'];
        vx = (en_x - st_x) / tl
        vy = (en_y - st_y) / tl
        # 5.2. Fit a straight line on the sequence
        pcnt = (seqstart + 1) % path_points_len;
        while pcnt != seqend:
            pl = pcnt - seqstart
            if pl < 0:pl = pl + path_points_len
            px = st_x + vx * pl;py = st_y + vy * pl
            ppcnt = path_pts[pcnt]
            pcpx = ppcnt['x'] - px;pcpy = ppcnt['y'] - py;
            dist2 = pcpx * pcpx + pcpy * pcpy
            if dist2 > ltres:curvepass = False
            if dist2 > errorval:errorpoint = pcnt;errorval = dist2;
            pcnt = (pcnt + 1) % path_points_len
        # return straight line if fits
        if curvepass:
            return Dict({
            'type': 'L',
            'x1': st_x + ofx,
            'y1': st_y + ofy,
            'x2': en_x + ofx,
            'y2': en_y + ofy
            })
        # 5.3. If the straight line fails (distance error>ltres), find the point with the biggest error
        fitpoint = errorpoint
        curvepass = True
        errorval = 0
        # 5.4. Fit a quadratic spline through this point, measure errors on every point in the sequence
        # helpers and projecting to get control point
        t = (fitpoint - seqstart) / tl
        t1 = (1 - t) * (1 - t)
        t2 = 2 * (1 - t) * t
        t3 = t * t
        cpx = (t1 * st_x + t3 * en_x - path_pts[fitpoint]['x']) / -t2
        cpy = (t1 * st_y + t3 * en_y - path_pts[fitpoint]['y']) / -t2
        # Check every point
        pcnt = seqstart + 1
        while pcnt != seqend:
            t = (pcnt - seqstart) / tl
            t1 = (1 - t) * (1 - t)
            t2 = 2 * (1 - t) * t
            t3 = t * t
            px = t1 * st_x + t2 * cpx + t3 * en_x
            py = t1 * st_y + t2 * cpy + t3 * en_y
            ppcnt = path_pts[pcnt]
            pcpx = ppcnt['x'] - px;pcpy = ppcnt['y'] - py;
            dist2 = pcpx * pcpx + pcpy * pcpy
            if dist2 > qtres:curvepass = False
            if dist2 > errorval:errorpoint = pcnt;errorval = dist2;
            pcnt = (pcnt + 1) % path_points_len
        # return spline if fits
        if curvepass == True:
            return Dict({
            'type': 'Q',
            'x1': st_x + ofx,'y1': st_y + ofy,
            'x2': cpx + ofx,'y2': cpy + ofy,
            'x3': en_x + ofx,'y3': en_y + ofy
            })
        # 5.5. If the spline fails (distance error>qtres), find the point with the biggest error
        splitpoint = fitpoint
        # Earlier: math.floor((fitpoint + errorpoint)/2);
        
        # 5.6. Split sequence and recursively apply 5.2. - 5.6. to startpoint-splitpoint and splitpoint-endpoint sequences
        self.split_point_ex = {'p':path, 'l': ltres, 'q': qtres, 'sp': splitpoint, 'en': seqend, 'is': True}
        return self.fitseq(path, ltres, qtres, seqstart, splitpoint)
        #return self.fitseq(path, ltres, qtres, seqstart, splitpoint).update( self.fitseq(path, ltres, qtres, splitpoint, seqend))



    # 5. Batch tracing paths
    def batchtracepaths(self,internodepaths, ltres, qtres):
        btracedpaths = []
        for k in internodepaths:
            if internodepaths.get(k, None) == None:continue
            btracedpaths.append(self.tracepath(internodepaths[k], ltres, qtres))
        return btracedpaths

    # 5. Batch tracing layers
    def batchtracelayers(self,binternodes, ltres, qtres):
        btbis = []
        for k in range(0,len(binternodes)):
            try:
                btbis.append( self.batchtracepaths(binternodes[k], ltres, qtres));
            except IndexError:
                print('Layer idx error was skipped!');continue
        return btbis

    #////////////////////////////////////////////////////////////
    #//
    #//  SVG Drawing functions
    #//
    #////////////////////////////////////////////////////////////

    def get_xy(self,d,num,ops,roundc):
        # options_roundcoords == -1: not use  roundtodec 0
        if roundc == -1:
            return d[f'x{num}'] * ops , d[f'y{num}'] * ops
        else:
            return self.roundtodec(d[f'x{num}'] * ops,1) , self.roundtodec(d[f'y{num}'] * ops,1)

    # Rounding to given decimals https://stackoverflow.com/questions/11832914/round-to-at-most-2-decimal-places-in-javascript
    #@lru_cache(maxsize=16)
    def roundtodec(self,val, places):
        #fmt = '{:.'+str(places)+'f}'
        return f'{{:.{places}f}}'.format(val)
    
    def cond_(self,obj,tr,el):
        if obj:return tr
        return el
    
    def svgpathstring(self,tracedata, lnum, pathnum, options):
        hcnt = 0;hsmp = 0;
        layer = tracedata['layers'][lnum]
        smp = layer[pathnum]
        smp_seg = smp['segments']
        smp_seg_max = len(smp_seg)
        ops = options['scale']
        closeZ = 'Z '
        
        # Filtering
        if (options['linefilter'] == True and (smp_seg_max < 3)):return '' # Line filter
        if tracedata['palette'][lnum]['a'] <= 10:return '' # Alpha filter
        if self.ignore_white == True and self.color_filter(tracedata['palette'][lnum]):return '' # Color filter
        if self.lasso_draw_mode == True:
            if tracedata['palette'][lnum]['a'] < 254:return ''
            tracedata['palette'][lnum] = self.fgc_rgb # Filled by Foreground Color
        #if self.open_path_mode == True:closeZ = ' '

        # print('Segment_Length',smp_seg_max)
        
        # Starting path element, desc contains layer and path number
        #str = '<path ' + self.cond_(options.desc,f'desc="l {lnum} p {pathnum}" ', '') + self.tosvgcolorstr(tracedata.palette[lnum], options) + 'd="'

        str = ''.join(['<path ',
                self.cond_(options['desc'],f'desc="l {lnum} p {pathnum}" ', ''),
                #self.tosvgcolorstr(tracedata['palette'][lnum]['r'],tracedata['palette'][lnum]['g'],tracedata['palette'][lnum]['b'],tracedata['palette'][lnum]['a'], options['strokewidth'],options['lineart']),
                self.tosvgcolorstr2(tracedata['palette'][lnum], options),
                'd="'
                ])

        # Creating non-hole path string
        # options_roundcoords == -1 is not use self.roundtodec()
        roundc = options['roundcoords']
        xx1,yy1 = self.get_xy(smp_seg[0],1,ops,roundc)
        str += f'M {xx1} {yy1} '

        for pcnt in range(0,smp_seg_max):
            #if closeZ == ' ' and pcnt == smp_seg_max-1:continue
            tp = smp_seg[pcnt]['type']
            xx2,yy2 = self.get_xy(smp_seg[pcnt],2,ops,roundc)
            str += f'{tp} {xx2} {yy2} '
            if smp_seg[pcnt].get('x3', None) != None:
                xx3,yy3 = self.get_xy(smp_seg[pcnt],3,ops,roundc)
                str += f'{xx3} {yy3} '
        str += closeZ

        # End of creating non-hole path string
        # Hole children
        #print("Sample:",layer[0])
        hcnt_max=len(smp['holechildren'])
        for hcnt in range(0,hcnt_max):
            #print("Check:",hcnt, "layer_length",len(layer))
            hsmp = layer[ smp['holechildren'][hcnt] ]
            hsmp_seg = hsmp['segments']
            hsmp_seg_total = len(hsmp_seg) - 1
            seglast = hsmp_seg[hsmp_seg_total]
            # workaround
            # print("Segment_length at hole", hsmp_seg_total)
            # options_roundcoords == -1 is not use self.roundtodec()
            roundc = options['roundcoords']
            if seglast.get('x3', None) != None:
                xx3,yy3 = self.get_xy(seglast,3,ops,roundc)
                str += f'M {xx3} {yy3} '
            else:
                xx2,yy2 = self.get_xy(seglast,2,ops,roundc)
                str += f'M {xx2} {yy2} '
            #pcnt = hsmp_seg_total
            for pcnt in range(hsmp_seg_total,-1,-1):# reversed(range(hsmp_seg_total+1)) hsmp_seg_total to 0
            # while pcnt >= 0:
                str += hsmp_seg[pcnt]['type'] + ' '
                if hsmp_seg[pcnt].get('x3', None) != None:
                    xx2,yy2 = self.get_xy(hsmp_seg[pcnt],2,ops,roundc)
                    str += f'{xx2} {yy2} '
                xx1,yy1 = self.get_xy(hsmp_seg[pcnt],1,ops,roundc)
                str += f'{xx1} {yy1} '
                # pcnt = pcnt - 1
            # End of creating hole path string
            str += closeZ# Close path
        # End of holepath check
        # Closing path element
        str += '" />'
        # Rendering control points
        opt_l = options['lcpr'];opt_q = options['qcpr'];
        
        if (opt_l or opt_q):
            qc = opt_q * 0.2
            lc = opt_l * 0.2
            pcnt_max = len(smp_seg)
            for pcnt in range(0,pcnt_max):
                if smp_seg[pcnt].get('x3', None) != None and opt_q:
                    xx2 = smp_seg[pcnt]['x2'] * ops;yy2 = smp_seg[pcnt]['y2'] * ops;
                    str += f'<circle cx="{xx2}" cy="{yy2}" r="{opt_q}" fill="cyan" stroke-width="{qc}" stroke="black" />'
                    
                    xx3 = smp_seg[pcnt]['x3'] * ops;yy3 = smp_seg[pcnt]['y3'] * ops;

                    str += f'<circle cx="{xx3}" cy="{yy3}" r="{opt_q}" fill="white" stroke-width="{qc}" stroke="black" />'
                    
                    xx1 = smp_seg[pcnt]['x1'] * ops;yy1 = smp_seg[pcnt]['y1'] * ops;
                    xx2 = smp_seg[pcnt]['x2'] * ops;yy2 = smp_seg[pcnt]['y2'] * ops;
                    
                    str += f'<line x1="{xx1}" y1="{yy1}" x2="{xx2}" y2="{yy2}" stroke-width="{qc}" stroke="cyan" />'
                    
                    xx2 = smp_seg[pcnt]['x2'] * ops;yy2 = smp_seg[pcnt]['y2'] * ops;
                    xx3 = smp_seg[pcnt]['x3'] * ops;yy3 = smp_seg[pcnt]['y3'] * ops;

                    str += f'<line x1="{xx2}" y1="{yy2}" x2="{xx3}" y2="{yy3}" stroke-width="{qc}" stroke="cyan" />'
                
                if smp_seg[pcnt].get('x3', None) == None and opt_l:
                    xx2 = smp_seg[pcnt]['x2'] * ops;yy2 = smp_seg[pcnt]['y2'] * ops;
                    str += f'<circle cx="{xx2}" cy="{yy2}" r="{opt_l}" fill="white" stroke-width="{lc}" stroke="black" />'
            # Hole children control points
            hcnt_max = len(smp['holechildren'])
            for hcnt in range(0,hcnt_max):
                hsmp = layer[smp['holechildren'][hcnt]]
                pcnt_max=len(hsmp_seg)
                for pcnt in range(0,pcnt_max):
                    hsp = hsmp_seg[pcnt]
                    if ((hsp.get('x3', None) != None) and opt_q):
                        xx2 = hsp['x2'] * ops;yy2 = hsp['y2'] * ops;
                        str += f'<circle cx="{xx2}" cy="{yy2}" r="{opt_q}" fill="cyan" stroke-width="{qc}" stroke="black" />'
                        
                        xx3 = hsp['x3'] * ops;yy3 = hsp['y3'] * ops;
                        str += f'<circle cx="{xx3}" cy="{yy3}" r="{opt_q}" fill="white" stroke-width="{qc}" stroke="black" />'
                        
                        xx1 = hsp['x1'] * ops;yy1 = hsp['y1'] * ops;
                        xx2 = hsp['x2'] * ops;yy2 = hsp['y2'] * ops;
                        str += f'<line x1="{xx1}" y1="{yy1}" x2="{xx2}" y2="{yy2}" stroke-width="{qc}" stroke="cyan" />'
                        
                        xx2 = hsp['x2'] * ops;yy2 = hsp['y2'] * ops;
                        xx3 = hsp['x3'] * ops;yy3 = hsp['y3'] * ops;
                        str += f'<line x1="{xx2}" y1="{yy2}" x2="{xx3}" y2="{yy3}" stroke-width="{qc}" stroke="cyan" />'
                        
                    if (hsp.get('x3', None) == None) and opt_l:
                        xx2 = hsp['x2'] * ops;yy2 = hsp['y2'] * ops;
                        str += f'<circle cx="{xx2}" cy="{yy2}" r="{opt_l}" fill="white" stroke-width="{lc}" stroke="black" />'
                    # End of  pcnt loop
                # End of hcnt loop
        # End of Rendering control points
        return str

    def getsvgstring (self,tracedata, options,flag):
        options = self.checkoptions(options)
        if (options['lineart'] == True) and (options['strokewidth'] <= 0): options['strokewidth'] = 2
        w = tracedata['width'] * options['scale']
        h = tracedata['height'] * options['scale']
        # SVG start
        #if options['viewbox']:
        #    vb = f'viewBox="0 0 {w} {h}" '
        #else:
        #    vb = f'width="{w}" height="{h}" '

        # Drawing: Layers and Paths loops
        # svghead = '<svg ' + vb + 'version="1.1" xmlns="http://www.w3.org/2000/svg" desc="Created with imagetracer.js version ' + self.versionnumber + '" >'
        svghead = '<svg>'
        svgstr=''
        svdbody = []
        svgbody_append = svdbody.append
        tls = tracedata['layers']
        lcnt =len(tls) - 1
        while lcnt >= 0 :
            pcnt =len(tls[lcnt])-1
            while pcnt >= 0:
                # Adding SVG <path> string
                if not tls[lcnt][pcnt]['isholepath']:
                    #svgstr = svgstr + self.svgpathstring(tracedata, lcnt, pcnt, options)
                    svgbody_append( self.svgpathstring(tracedata, lcnt, pcnt, options) )
                pcnt = pcnt - 1
            # End of paths loop
            lcnt = lcnt - 1
        # End of layers loop
        # SVG End
        if flag == False:return ''.join(svdbody)
        return ''.join([svghead , ''.join(svdbody) , '</svg>'])


    # Comparator for numeric Array.sort
    def compareNumbers(self,a, b):
        return a - b

    # Convert color object to rgba string
    def torgbastr(self,c):
        return f'''rgba({c['r']},{c['g']},{c['b']},{c.a})'''


    def color_filter(self,c):
        # message( f'''rgba({c['r']},{c['g']},{c['b']},{c.a})''')
        if c['r'] == 255 and c['g'] == 255 and c['b'] == 255 :return True
        else:return False


    def tosvgcolorstr(self,r,g,b,a,sw,ld):
        # sw = options['strokewidth'],  ld = options['lineart']
        op = a / 255.0
        rgbcolor = f'rgb({r},{g},{b})'
        if ld == True:
            return f'fill="none" stroke="{rgbcolor}" stroke-width="{sw}"  paint-order="stroke" stroke-linejoin="round" opacity="{op}" '
        return f'fill="{rgbcolor}" stroke="{rgbcolor}" stroke-width="{sw}"  paint-order="stroke" stroke-linejoin="round" opacity="{op}" '


    # Convert color object to SVG color string (original ver)
    def tosvgcolorstr2(self,c,options):
        sw = options['strokewidth'];ld = options['lineart'];
        op = c['a'] / 255.0
        if self.lasso_draw_mode == True:
            s = self.bgc_rgb;
            strkcolor = f'''rgb({s['r']},{s['g']},{s['b']})'''
            fillcolor = f'''rgb({c['r']},{c['g']},{c['b']})'''
        else:
            strkcolor = f'''rgb({c['r']},{c['g']},{c['b']})''';
            fillcolor = f'''rgb({c['r']},{c['g']},{c['b']})'''
        if ld == True:
            return f'fill="none" stroke="{strkcolor}" stroke-width="{sw}"  paint-order="stroke" stroke-linejoin="round" opacity="{op}" '
        return f'fill="{fillcolor}" stroke="{strkcolor}" stroke-width="{sw}"  paint-order="stroke" stroke-linejoin="round" opacity="{op}" '



    # HTML Helper function: Appending an <svg> element to a container from an svgstring
    def appendSVGString(svgstr, parentid):
        return

    #////////////////////////////////////////////////////////////
    #//
    #//  Canvas functions
    #//
    #////////////////////////////////////////////////////////////


    def blur(self,imgd, radius, delta):
        i,j,k,d,idx = 0,0,0,0,0
        racc,gacc,bacc,aacc,wacc = 0,0,0,0,0
        w = imgd['width']
        h = imgd['height']
        idata = imgd['data']
        empdata = [0] * len(idata)
        # new ImageData
        imgd2 = Dict({
            'width': w,
            'height': h,
            'data': empdata.copy()
            })
        
        imgd2_data = empdata.copy()
        
        # radius and delta limits, this kernel
        radius = math.floor(radius)
        if radius < 1:return imgd
        if radius > 5:radius = 5
        delta = abs(delta)
        if delta > 1024:delta = 1024
        # Gaussian Table
        thisgk = self.gks[radius - 1]
        # loop through all pixels, horizontal blur
        for j in range(0,h):
            for i in range(0,w):
                # gauss kernel loop
                for k in range(-radius,radius + 1):
                    # add weighted color values
                    if (i + k > 0) and (i + k < w):
                        idx = (j * w + i + k) * 4
                        tgk = thisgk[k + radius];
                        # print(imgd.data[idx] , tgk)
                        imgq = idata[idx:idx+4]# get a list of imgd.data[idx] to imgd.data[idx + 3]
                        racc = racc + imgq[0] * tgk
                        gacc = gacc + imgq[1] * tgk
                        bacc = bacc + imgq[2] * tgk
                        aacc = aacc + imgq[3] * tgk
                        wacc = wacc + tgk
                # The new pixel
                idx = (j * w + i) * 4
                imgd2_data[idx] = floor(racc / wacc)
                imgd2_data[idx + 1] = floor(gacc / wacc)
                imgd2_data[idx + 2] = floor(bacc / wacc)
                imgd2_data[idx + 3] = floor(aacc / wacc)
            # End of width loop
        # End of horizontal blur
        # copying the half blurred imgd2 # Uint8,  ClampedArray?
        # himgd = new Uint8ClampedArray(imgd2.data)
        himgd = imgd2_data # empdata.copy()
        
        # loop through all pixels, vertical blur
        for j in range(0,h):
            for i in range(0,w):
                # gauss kernel loop
                for k in range(-radius,radius + 1):
                    # add weighted color values
                    if j + k > 0 and j + k < h:
                        idx = ((j + k) * w + i) * 4
                        tgk = thisgk[k + radius]
                        himgq = himgd[idx:idx+4]# get a list of himgd[idx] to himgd[idx + 3]
                        racc = racc + himgq[0] * tgk
                        gacc = gacc + himgq[1] * tgk
                        bacc = bacc + himgq[2] * tgk
                        aacc = aacc + himgq[3] * tgk
                        wacc = wacc + tgk
                # The new pixel
                idx = (j * w + i) * 4
                imgd2_data[idx] = floor(racc / wacc)
                imgd2_data[idx + 1] = floor(gacc / wacc)
                imgd2_data[idx + 2] = floor(bacc / wacc)
                imgd2_data[idx + 3] = floor(aacc / wacc)
                # End of width loop
        # End of vertical blur
        # Selective blur: loop through all pixels
        for j in range(0,h):
            for i in range(0,w):
                idx = (j * w + i) * 4
                # d is the difference between the blurred and the original pixel
                imgq = idata[idx:idx+4]# get a list of imgd.data[idx] to imgd.data[idx + 3]
                img2q = imgd2_data[idx:idx+4]
                d = abs(img2q[0] - imgq[0]) + abs(img2q[1] - imgq[1]) + abs(img2q[2] - imgq[2]) + abs(img2q[3] - imgq[3])
                # selective blur: if d>delta, put the original pixel back
                if d > delta:
                    imgd2_data[idx] = imgq[0]
                    imgd2_data[idx + 1] = imgq[1]
                    imgd2_data[idx + 2] = imgq[2]
                    imgd2_data[idx + 3] = imgq[3]
                # End of i loop
            # End of j loop
        # End of Selective blur
        imgd2.data = imgd2_data
        return imgd2

    #////////////////////////////////////////////////////////////
    #//
    #//  Docker functions
    #//
    #////////////////////////////////////////////////////////////


d_opt=Dict()
x_opt={
    'ignore_white' : False, 
    'lasso_draw_mode' : False,
    'open_path_mode' : False,
    'gray_mode' : False,
    'warning_skip': False,
    'continuous_draw': True
    }

chkb_ext = None
chkb_ext2 = None
chkb_ext3 = None
chkb_ext4 = None
chkb_cond = None
opt_keys=[]

pmenu = Dict({
    'Make a palette': 0,
    'Random': 1,
    'Deterministic': 2
    })

lmenu = Dict({
    'Sequential': 0,
    'Parallel (Fast)': 1
    })

 
# Custom Widget (Horizontal Line,Vertical Line)
class QHLine(QWidget):
    def __init__(self):
        super().__init__()
        self.initWidget()
        
    def initWidget(self):
        self.hl = QFrame()
        self.hl.setFrameShape(QFrame.HLine)
        self.hl.setFrameShadow(QFrame.Sunken)
        layout = QHBoxLayout()
        layout.addWidget(self.hl)
        layout.setContentsMargins(0,0,0,0)
        self.setLayout(layout) # It nessesasry

class QVLine(QWidget):
    def __init__(self):
        super().__init__()
        self.initWidget()
        
    def initWidget(self):
        self.hl = QFrame()
        self.hl.setFrameShape(QFrame.VLine)
        self.hl.setFrameShadow(QFrame.Sunken)
        layout =QVBoxLayout()
        layout.addWidget(self.hl)
        layout.setContentsMargins(0,0,0,0)
        self.setLayout(layout) # It nessesasry


class Vectrize(DockWidget):

    def __init__(self):
        global x_opt,opt,opt_keys,pmenu,lmenu,chkb_ext,chkb_ext2,chkb_ext3,chkb_ext4,chkb_cond,plugin_ver
        super().__init__()
        self.setWindowTitle(plugin_ver)
        self.opt=Dict()
        gen = ImageToSVGConverter()
        opt_keys = list(gen.optionpresets['default'].keys())
        data_option_presets = gen.optionpresets['default']
        
        widget = QWidget()
        btn = QPushButton("Vectrize!",self)
        
        
        vbox = QVBoxLayout()
        hbox = QHBoxLayout()
        fboxz = QFormLayout()
        fbox = QFormLayout()

        hbox3 = QHBoxLayout()
        hbox4 = QHBoxLayout()
        d_opt.corsenabled = False
        d_opt.viewbox = True
        d_opt.desc = False
        
        d_opt.colorsampling = QComboBox()
        d_opt.layering = QComboBox()

        chbox = Dict()

        hl0 = QHLine()
        hl = QHLine()
        hl2 = QHLine()
        hl3 = QHLine()
        hl4 = QHLine()
        hl5 = QHLine()
        
        hv0 = QVLine()
        hv1 = QVLine()
        
        # Combo box setting
        pkey= list(pmenu.keys())
        for pm in pkey:
            d_opt.colorsampling.addItem(pm);
        d_opt.colorsampling.setCurrentIndex(data_option_presets['colorsampling'])

        # Combo box setting2
        lkey= list(lmenu.keys())
        for lm in lkey:
            d_opt.layering.addItem(lm);
        d_opt.layering.setCurrentIndex(data_option_presets['layering'])


        d_opt.ltres = QLineEdit(str(data_option_presets['ltres']))
        d_opt.qtres = QLineEdit(str(data_option_presets['qtres']))
        d_opt.pathomit = QLineEdit(str(data_option_presets['pathomit']))

        d_opt.numberofcolors = QLineEdit(str(data_option_presets['numberofcolors']))
        d_opt.mincolorratio = QLineEdit(str(data_option_presets['mincolorratio']))
        d_opt.colorquantcycles = QLineEdit(str(data_option_presets['colorquantcycles']))


        d_opt.strokewidth = QLineEdit(str(data_option_presets['strokewidth']))

        d_opt.scale = QLineEdit(str(data_option_presets['scale']))
        d_opt.roundcoords = QLineEdit(str(data_option_presets['roundcoords']))


        d_opt.rightangleenhance = QCheckBox(" ",self) # bool
        d_opt.rightangleenhance.setChecked( data_option_presets['rightangleenhance']) # bool_chk

        d_opt.linefilter = QCheckBox("",self); # bool
        d_opt.linefilter.setChecked( data_option_presets['linefilter'] )  # bool_chk

        d_opt.lineart = QCheckBox("",self) # bool
        d_opt.lineart.setChecked( data_option_presets['lineart'] )  # bool_chk
        
        d_opt.blurradius = QLineEdit(str(data_option_presets['blurradius']))
        d_opt.blurdelta = QLineEdit(str(data_option_presets['blurdelta']))


        f_cont0 = QHBoxLayout()
        f_lt0 = QFormLayout()
        f_rt0 = QFormLayout()

        es = QLabel('Error threshold for');es.setFont(QFont('Arial',12,QFont.DemiBold));
        ltr = QLabel('Line')
        ltr.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        ltr.setToolTip('Set Large/Small value for large/small image size')
        f_lt0.addRow(ltr,d_opt.ltres)
        
        qtr = QLabel('Q-Spline')
        qtr.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        qtr.setToolTip('Set Large/Small value for large/small image size')
        f_rt0.addRow(qtr,d_opt.qtres);

        f_cont0.addLayout(f_lt0)
        f_cont0.addLayout(f_rt0)


        ucol = QLabel('Use Colors ')
        ucol.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        ucol.setToolTip('1 - 7 : GrayScale \n 8 - : RGBColor)')
        fboxz.addRow(ucol,d_opt.numberofcolors)
        fboxz.addRow(QLabel('Color Sampling'),d_opt.colorsampling) # combo
        fboxz.addRow(QLabel('Layering Method'),d_opt.layering) # combo

        
        f_cont = QHBoxLayout()
        f_lt = QFormLayout()
        f_rt = QFormLayout()

        el = QLabel('Color Options');el.setFont(QFont('Arial',12,QFont.DemiBold));
        fl = QLabel('Settings');fl.setFont(QFont('Arial',12,QFont.DemiBold));

        # left column
        f_lt.addRow(el,QLabel(''));f_rt.addRow(fl,QLabel(''));

        mcr = QLabel('Min Color Ratio')
        mcr.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        mcr.setToolTip(' If (Total pixels * Min color Ratio)  &lt; Colors,use random color')
        f_lt.addRow(mcr,d_opt.mincolorratio)


        cqc = QLabel('Quantize Cycle')
        cqc.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        cqc.setToolTip('Color quantize times (1–3) \n Larger values makes slower ')
        f_lt.addRow(cqc,d_opt.colorquantcycles)

        chkb_ext = QCheckBox("",self) # bool
        chkb_ext.setChecked( x_opt['ignore_white'] )
        f_lt.addRow(QLabel('Ignore white pixel'),chkb_ext);# bool

        asv = QLabel('Shape Support Tools');asv.setFont(QFont('Arial',12,QFont.DemiBold));
        size_policy = QSizePolicy(QSizePolicy.Minimum,QSizePolicy.Minimum)

        # Color picker
        btn2 = QPushButton("Color Picker",self)
        btn2.setIcon(Krita.instance().icon('sample-screen'))
        btn2.clicked.connect(self.scrn_smpl)
        btn2.setToolTip('Color Picker:\nDirect change the shape color\n Useful for shape edit or color matching \n This triggered to Sample Screen command')

        # rid of sawtooth
        btn3 = QPushButton("Rid of SawTooth",self)
        btn3.setIcon(Krita.instance().icon('roundmarker'))
        btn3.clicked.connect(self.erode)
        btn3.setToolTip('Use to selection before vectrize,\n It become smooth. \n This triggered to smooth command x4')

        # shape to selection
        btn4 = QPushButton("Shape to Sel",self)
        btn4.setIcon(Krita.instance().icon('import-as-selectionMask'))
        btn4.clicked.connect(self.shp_sel)
        btn4.setToolTip('Make selection from Selected Shape\n This triggered to convert shapes to vector selection')

        # raise
        btn5 = QPushButton("",self)
        btn5.setIcon(Krita.instance().icon('arrow-up'))
        btn5.setMaximumSize(40,40)
        btn5.setSizePolicy(size_policy)
        btn5.clicked.connect(self.shp_raise)
        btn5.setToolTip('Raise the shape \nRaise order the shape that current selected ')

        # down
        btn6 = QPushButton("",self)
        btn6.setIcon(Krita.instance().icon('arrow-down'))
        btn6.setMaximumSize(40,40)
        btn6.setSizePolicy(size_policy)
        btn6.clicked.connect(self.shp_lower)
        btn6.setToolTip('Down the shape \nDown order the shape that current selected ')

        # style copy
        btn7 = QPushButton("",self)
        btn7.setIcon(Krita.instance().icon('similar-selection'))
        btn7.setMaximumSize(40,40)
        btn7.setSizePolicy(size_policy)
        btn7.clicked.connect(self.shp_cpy)
        btn7.setToolTip('Copy Shape Style:\n Copy style of shape that current selected\n Stroke color and width,Fill color etc')

        # style paste
        btn8 = QPushButton("",self)
        btn8.setIcon(Krita.instance().icon('download'))
        btn8.setMaximumSize(40,40)
        btn8.setSizePolicy(size_policy)
        btn8.clicked.connect(self.shp_style)
        btn8.setToolTip('Paste Shape Style:\n Paste style to current selected shape\n Stroke color and width,Fill color_q_cycles etc')



        v_ord = QVBoxLayout()
        v_ord.addWidget(btn5);v_ord.addWidget(btn6);
        
        v_ord2 = QVBoxLayout()
        v_ord2.addWidget(btn7);v_ord2.addWidget(btn8);

        v_ord3 = QVBoxLayout()
        t1 = QHBoxLayout()
        t2 = QHBoxLayout()
        t1.addWidget(asv);t1.addWidget(btn2);
        t2.addWidget(btn3);t2.addWidget(btn4);
        
        v_ord3.addLayout(t1);v_ord3.addLayout(t2);
        hbox4.addLayout(v_ord);hbox4.addLayout(v_ord2);hbox4.addLayout(v_ord3);

        # chkb_ext4 = QCheckBox("GrayScale",self) # bool
        # chkb_ext4.setChecked( x_opt['gray_mode'] )

        # right column
        
        f_rt.addRow(QLabel('Scale'),d_opt.scale)
        
        pom = QLabel('Path omit');
        pom .setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        pom .setToolTip('Path omit shorter than tihs value.(0–25)')
        f_rt.addRow(pom,d_opt.pathomit)
        
        dcm = QLabel('Rounding dec.pts')
        dcm .setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        dcm .setToolTip('Rounding number of decimal point')
        f_rt.addRow(dcm,d_opt.roundcoords)

        rae = QLabel('90° Enhance')
        rae.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        rae.setToolTip('Enhance Right Angle(90 degree) coner')
        f_rt.addRow(rae,d_opt.rightangleenhance) # bool

        f_rt.addRow(QLabel('Line remove filter'),d_opt.linefilter);# bool

        f_cont.addLayout(f_lt)
        f_cont.addWidget(hv0)
        f_cont.addLayout(f_rt)


        # Form Box2
        bl = QLabel('Blur');bl.setFont(QFont('Arial',12,QFont.DemiBold));
        lst = QLabel('Stroke (Line)');lst.setFont(QFont('Arial',12,QFont.DemiBold));

        f_cont2 = QHBoxLayout()
        f_lt2 = QFormLayout()
        f_rt2 = QFormLayout()



        f_lt2.addRow(bl,QLabel(''));
        f_lt2.addRow(QLabel('Radius (0–5)'),d_opt.blurradius)
        f_lt2.addRow(QLabel('Delta (&lt;1024)'),d_opt.blurdelta)

        f_rt2.addRow(lst,QLabel(''));
        f_rt2.addRow(QLabel('Stroke Only mode'),d_opt.lineart);# bool
        f_rt2.addRow(QLabel('Stroke width'),d_opt.strokewidth)


        f_cont2.addLayout(f_lt2)
        f_cont2.addWidget(hv1)
        f_cont2.addLayout(f_rt2)

        # hbox3.addRow(QLabel('Viewbox'),self.opt_viewbox);d_opt.viewbox = self.opt_viewbox.isChecked() # bool
        # self.opt_desc = QCheckBox("Add desc attribute");self.opt_desc.setChecked( data_option_presets['desc'] ) # bool
        # hbox3.addRow(QLabel('desc info'),self.opt_desc);d_opt.desc = self.opt_desc.isChecked() # bool_chk


        
        d_opt.lcpr = data_option_presets['lcpr']
        d_opt.qcpr = data_option_presets['qcpr']

        chkb_ext2 = QPushButton()# bool button
        chkb_ext2.setChecked( x_opt['lasso_draw_mode'] )
        chkb_ext2.setCheckable(True)
        chkb_ext2.setIcon(Krita.instance().icon('tool_outline_selection'))
        chkb_ext2.setIconSize(QSize(16, 16))
        chkb_ext2.setText("Lasso Mode")
        chkb_ext2.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        chkb_ext2.setToolTip('Use freehand selection tool,and Fill color as Fore GroundColor \n Stroke(Border) color as BackGround Color')
        chkb_ext2.clicked.connect(lambda: self.on_lasso_mode_toggled(chkb_ext2.isChecked()))

        chkb_ext3 = QCheckBox("Skip the warning.",self) # bool
        chkb_ext3.setChecked( x_opt['warning_skip'] )
        chkb_ext3.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        chkb_ext3.setToolTip('Skip the dialogs that \n Image size warning and Elapse Time')

        chkb_cond = QCheckBox("Keep active layer");
        chkb_cond.setChecked( x_opt['continuous_draw'] ) # bool
        chkb_cond.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        chkb_cond.setToolTip('Keep current layer focus for continuous drawing')



        vlast = QHBoxLayout()
        vlast.addWidget(chkb_ext2)
        vlast.addWidget(chkb_cond)
        vlast.addWidget(chkb_ext3)
        
        hlast = QHBoxLayout()
        hlast.addWidget(btn)

        #Layout
        vbox.addWidget(es)
        vbox.addLayout(f_cont0)
        vbox.addWidget(hl0)
        vbox.addLayout(fboxz)
        vbox.addWidget(hl)
        vbox.addLayout(f_cont)
        vbox.addWidget(hl2)
        vbox.addLayout(f_cont2)
        vbox.addWidget(hl5)

        vbox.addLayout(hbox4)
        
        vbox.addLayout(hbox)
        vbox.addWidget(hl3)
        
        vbox.addLayout(vlast)
        vbox.addLayout(hlast)
        
        ft=QLabel(' Powerd by Imagetracer.js 1.2.6')
        ft.setFont(QFont('Arial',10)) 
        ft.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignVCenter)
        vbox.addWidget(ft)
        tbox = QHBoxLayout()
        tbox.addSpacing(8)
        tbox.addLayout(vbox)
        tbox.addSpacing(8)
        widget.setLayout(tbox)
        self.setWidget(widget)
        btn.setIcon(Krita.instance().icon('split-layer'))
        btn.clicked.connect(self.vector_exe)
        
    def canvasChanged(self, canvas):
        pass


    def ch_toggle(self, toggle):
        if toggle == QtCore.Qt.Checked:
            return True
        else:
            return False

    def scrn_smpl(self, canvas):
        Krita.instance().action('sample_screen_color').trigger()

    def scrn_smpl(self, canvas):
        Krita.instance().action('sample_screen_color').trigger()

    def shp_sel(self, canvas):
        Krita.instance().action('convert_shapes_to_vector_selection').trigger()

    def shp_raise(self, canvas):
        Krita.instance().action('object_order_raise').trigger()

    def shp_lower(self, canvas):
        Krita.instance().action('object_order_lower').trigger()

    def shp_cpy(self, canvas):
        Krita.instance().action('edit_copy').trigger()


    def shp_style(self, canvas):
        Krita.instance().action('paste_shape_style').trigger()


    def erode(self, canvas):
        Krita.instance().action('smoothselection').trigger()
        Krita.instance().action('smoothselection').trigger()
        Krita.instance().action('smoothselection').trigger()
        Krita.instance().action('smoothselection').trigger()

    def on_lasso_mode_toggled(self, checked):
        if checked:
            Krita.instance().action('KisToolSelectOutline').trigger()
            chkb_ext3.setChecked(True)
            chkb_cond.setChecked(True)
        else:
            pass

    # called after setup(self)
    def vector_exe(self):
        global opt,opt_keys
        sender = self.sender()
        # User input parameters -> Priset data, Created and add preset data
        m = Dict()
        opt = d_opt
        keys = opt_keys
        
        
        for k in keys:
            # Separation str,bool -> float,int,bool -> into preset
            if opt[k] == {}:continue
            
            if (k == 'colorsampling') or (k == 'layering'):# combobox
                optdata = 0
                if k == 'colorsampling':
                    optdata = int(pmenu[opt[k].currentText()])
                if k == 'layerring':
                    optdata = int(lmenu[opt[k].currentText()])
                # message(str(opt[k].currentText())+str(optdata))
                m['krita_set'][k] = optdata
                continue


            if type(opt[k]) == int or type(opt[k]) == float:# This is not from QLineEdit
                optdata = opt[k]
                if optdata < 0.0: optdata = 0.0
                if optdata < 0: optdata = 0
                m['krita_set'][k] = optdata
                continue

            if type(opt[k]) == bool: # raw bool data
                m['krita_set'][k] = opt[k]
                continue

            if (k == 'lineart') or (k == 'rightangleenhance') or (k == 'linefilter'): # raw bool This from Qcheckbox
                optdata = (opt[k].checkState() == QtCore.Qt.Checked)
                m['krita_set'][k] = optdata
                continue


            if type(opt[k].text()) == str:
                optdata = opt[k].text()# String = This from QLineEdit 
                if optdata == "" or  optdata == None:optdata = 0
                # Separation float,int
                try:
                    if '.' in optdata:
                        optdata = float(optdata)
                        if optdata < 0.0: optdata = 0.0
                        if optdata > 1024.0: optdata = 1024.0
                    else:
                        optdata = int(optdata)
                        if optdata < 0: optdata = 0
                        if optdata > 1024: optdata = 1024
                except Exception as e:
                    optdata = 0
                    message(f'Invalid value detected\n Exception:{e}')
                    return
                m['krita_set'][k] = optdata
            # etc
            else:
                continue
        # message(m)
        krita_instance = Krita.instance()
        currentDoc = krita_instance.activeDocument()
        currentWin = krita_instance.activeWindow().qwindow()
        
        try:
            target_krita_node = currentDoc.activeNode()
        except:
            return
            



        if target_krita_node.type() != 'paintlayer':
            message('This effect to only Paint Layer,\n Please select a Paint Layer!')
            return
            

        gen = ImageToSVGConverter()
        
        # gen.optionpresets.update(m['krita_set'])
        """
        message(
            "lineart:"+str(m['krita_set']['lineart'])+"_"+
            "Filt:"+str(m['krita_set']['linefilter'])+"_"+
            "Angle:"+str(m['krita_set']['rightangleenhance'])
        )
        """
        gen.optionpresets['default'] = m['krita_set']
        gen.ignore_white = chkb_ext.isChecked()
        gen.lasso_draw_mode = chkb_ext2.isChecked()
        gen.continuous_draw = chkb_cond.isChecked()
        gen.warning_skip = chkb_ext3.isChecked()
        # gen.gray_mode = chkb_ext4.isChecked()
        
        w = currentDoc.width();h = currentDoc.height();
        if gen.warning_skip == False and (w > 756 or h > 756):
            ans = message_yn(f'Image size({w}px*{h}px) is too large,\n Long time (60sec or more) left if convert whole image probably.\n Do you want to cancel? ')
            if ans == False:return
        st = time.time()
        
        gen.kritaToSVG()
        
        elapse = time.time() - st
        if gen.warning_skip == False:
            message(f'Elapse time:{elapse}')



# https://doc.qt.io/qtforpython-5/PySide2/QtWidgets/QMessageBox.html
# This function is useful for assert debug message when can't use scripter even.
def message(mes):
    mb = QMessageBox()
    mb.setText(str(mes))
    mb.setStandardButtons(QMessageBox.Ok)
    ret = mb.exec()
    if ret == QMessageBox.Ok:
        pass # OK clicked


def message_yn(mes):
    mb = QMessageBox()
    mb.setText(str(mes))
    mb.setWindowTitle('Warning!')
    mb.setStandardButtons(QMessageBox.Ok|QMessageBox.Cancel)
    ret = mb.exec()
    if ret == QMessageBox.Ok:
        return True # OK clicked
    if ret == QMessageBox.Cancel:
        return False # Cancel clicked

