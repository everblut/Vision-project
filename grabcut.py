#!/usr/bin/env python
'''
===============================================================================
Algoritmo grabcut

USO:
    python grabcut.py <filename>
    
    formatos permitidos : [.jpeg] [.jpg] [.png] [.bmp]

GUIA:
    En la ventana de input debera marcarse un rectangulo (usando click derecho)
    para iniciar con la segmentacion. Utilizando el boton derecho y despues
    usando la tecla {s} para hacer actualizar la segmentacion de la imagen.

    La tecla {v} iniciara el modo de marcador verdadero para identificar 
    segmentos que si van dentro de la imagen, usando dicha tecla y despues
    el click izquierdo se puede marcar que partes si son del nuevo corte.

    La tecla {f} iniciara el modo de marcador falso para identificar
    segmentos que no van dentro de la imagen, usando dicha tecla y despues
    el click izquierdo se puede marcar las partes que no son del nuevo corte.

    Usando la tecla {g} se puede guardar el corte en una nueva imagen.
    (Guardara el output actual)

    Se puede reiniciar el algoritmo usando la tecla {r}


===============================================================================
'''

from PIL import Image
import numpy as np
import cv2
import sys
import time
import random
import math

BLUE = [255,0,0]        # Color del rectangulo
BLACK = [0,0,0]         # marcador falso
WHITE = [255,255,255]   # marcador verdadero

DRAW_BG = {'color' : BLACK, 'val' : 0}
DRAW_FG = {'color' : WHITE, 'val' : 1}

# flags
rect = (0,0,1,1)
drawing = False         # flag para saber cuando se dibujara
rectangle = False       # flag para saber cuando se usara el rectangulo
rect_over = False       # flag para saber si estan dibujando un rectangulo
rect_or_mask = 100      # flag para saber si usan el rectangulo o una mascara
value = DRAW_FG         # primero se utiliza el trazo verdadero
thickness = 3           # que tan fino es el punto de dibujo

def onmouse(event,x,y,flags,param):
    global img,img2,drawing,value,mask,rectangle,rect,rect_or_mask,ix,iy,rect_over

    # Dibujar rectangulo
    if event == cv2.EVENT_RBUTTONDOWN:
        rectangle = True
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if rectangle == True:
            img = img2.copy()
            cv2.rectangle(img,(ix,iy),(x,y),BLUE,2)
            rect = (min(ix,x),min(iy,y),abs(ix-x),abs(iy-y))
            rect_or_mask = 0

    elif event == cv2.EVENT_RBUTTONUP:
        rectangle = False
        rect_over = True
        cv2.rectangle(img,(ix,iy),(x,y),BLUE,2)
        rect = (min(ix,x),min(iy,y),abs(ix-x),abs(iy-y))
        rect_or_mask = 0
        print " Actualiza la imagen para mayor precision tecla {a}\n"

    # dibujando los puntos

    if event == cv2.EVENT_LBUTTONDOWN:
        if rect_over == False:
            print "Primero selecciona una region para segmentar\n"
        else:
            drawing = True
            cv2.circle(img,(x,y),thickness,value['color'],-1)
            cv2.circle(mask,(x,y),thickness,value['val'],-1)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.circle(img,(x,y),thickness,value['color'],-1)
            cv2.circle(mask,(x,y),thickness,value['val'],-1)

    elif event == cv2.EVENT_LBUTTONUP:
        if drawing == True:
            drawing = False
            cv2.circle(img,(x,y),thickness,value['color'],-1)
            cv2.circle(mask,(x,y),thickness,value['val'],-1)

def normalizar(image, max_values):
    pic = image.load()
    for i in range(image.size[0]):
        for j in range(image.size[1]):
            (R, G, B) = pic[i,j]
            if R != 0:
                R = ((float(R)/max_values[0])*255 )
            if G != 0:
                G = ((float(G)/max_values[1])*255)
            if B != 0:
                B = ((float(B)/max_values[2])*255)
            R = G = B = int( ( R + G + B )/3 )
            pic[i,j] = (R, G, B)
                
def gradiente(gradientx, gradienty):
    gx = gradientx.load()
    gy = gradienty.load()
    max_values = [0,0,0]
    for i in range(gradientx.size[0]):
        for j in range(gradientx.size[1]):
            (Rx, Gx, Bx) = gx[i,j]
            (Ry, Gy, By) = gy[i,j]
            Rx = int(math.sqrt(Rx+Ry))
            if Rx > max_values[0]:
                max_values[0] = Rx
                Gx = int(math.sqrt(Gx+Gy))
            if Gx > max_values[1]:
                max_values[1] = Gx
                Bx = int(math.sqrt(Bx+By))
            if Bx > max_values[2]:
                max_values[2] = Bx
                gx[i,j] = (Rx, Gx, Bx)
    return gradientx, max_values
                
                
def detectar_borde(picture, output="g"):
    imagex = aplicar_mascara(picture, ["sobelx"], [1.0/1.0], cmd="i")
    imagey  = aplicar_mascara(picture, ["sobely"], [1.0/1.0], cmd="i")
    border, max_values = gradient(imagex, imagey)
    del imagey
    
    pseudo_promedio = self.normalizar(border, max_values)
    filtro_umbral(border, umbral=80)
    
    border.save(output)
        
def copiar_matriz(matrix):
    new = list()
    for i in matrix:
        temp = list()
        for j in i:
            temp.append(j)
        new.append(temp)
    return new
        
def convolucion(kernel, image):
    pic = image.load()
    pic_copy = (image.copy()).load()
    res = dict()
    for i in range(image.size[0]):
        for j in range(image.size[1]):
            
            sumatory = [0.0, 0.0, 0.0] # RGB
            kernel_len = len(kernel[0])
            kernel_pos = 0
            
            for h in range(i-1, i+2):
                for l in range(j-1, j+2):
                    if h >= 0 and l >= 0 and h < image.size[0] and l < image.size[1]:
                        pixel = pic_copy[h, l]
                        sumatory[0] += pixel[0]*kernel[int(kernel_pos/3)][kernel_pos%3]
                        sumatory[1] += pixel[1]*kernel[int(kernel_pos/3)][kernel_pos%3]
                        sumatory[2] += pixel[2]*kernel[int(kernel_pos/3)][kernel_pos%3]
                        kernel_pos += 1
            res[i, j] = (int(sumatory[0]), int(sumatory[1]), int(sumatory[2]))
    return res
        
def multiplicar_mascara(kernel, const):
    for i in range(len(kernel)):
        for j in range(len(kernel[0])):
            kernel[i][j] *= const
    return kernel
        
def aplicar_mascara(nImagen, mascara=[], const=[], nOutput="output.jpg", image=None, cmd=""):
    if image == None:
        image = Image.open(nImagen)
    for i in range(len(mascara)):
        kernel = self.matrix_copy(self.indice_mascara[mascara[i]])
        kernel = self.multiplicar_mascara(kernel, const[i])
        res = self.convolucion(kernel, image)
    
    if cmd == "i":
        return res
    else:
        pass
            
def filtro_umbral(image, umbral=128):
    pic = image.load()
    for i in range(image.size[0]):
        for j in range(image.size[1]):
            
            colors = list(pic[i,j])
            for h in range(len(colors)):
                if colors[h] < umbral:
                    colors[h] = 0
                else:
                    colors[h] = 255
            pic[i,j] = tuple(colors)
    return pic

print __doc__

# cargar imagen
if len(sys.argv) == 2:
    filename = sys.argv[1] 
else:
    print "No se ingreso imagen, usando una por default\n"
    filename = 'FIME.jpg'

formatos = ['jpeg','jpg','png','bmp']
if filename.split('.')[1] not in formatos:
    print "Solo se soportan los formatos \n[.jpeg],[.jpg],[.png],[.bmp] utilize alguno de ellos."
    sys.exit(0)

img = cv2.imread(filename)
if img is None:
    print "Err: depth, el formato no tiene 3 canales+alpha \nNo se pudo abrir la imagen, verifique el formato"
    sys.exit(0)

img2 = img.copy()                               # realizamos una copia
mask = np.zeros(img.shape[:2],dtype = np.uint8) # se usa una mascara de ceros
output = np.zeros(img.shape,np.uint8)           # imagen resultado
h,w,d = img.shape

if h < 100 or w < 100:
    print "Tu imagen mide %sx%s \nIngresa una imagen de por lo menos 100x100"%(h,w)
    sys.exit(0)

if h > 900 or w > 900:
    print "Tu imagen mide %sx%s \nIngresa una imagen a lo maximo de 900 x 900"%(w,h)
    sys.exit(0)

# ventanas
cv2.namedWindow('salida')
cv2.namedWindow('entrada')
cv2.setMouseCallback('entrada',onmouse)
cv2.moveWindow('entrada',img.shape[1]+10,90)
print " Primero debe dibujar un rectangulo sobre el objeto usado click derecho: \n"

while(1):

    cv2.imshow('salida',output)
    cv2.imshow('entrada',img)
    k = 0xFF & cv2.waitKey(1)
    if k == 27:         # ESC para salir
        break
    elif k == ord('f'): # modo falso
        print " marca las regiones que no forman parte del objeto \n"
        value = DRAW_BG
    elif k == ord('v'): # modo verdadero
        print "[Modo verdadero] marca las regiones que si forman parte del objeto \n"
        value = DRAW_FG
    elif k == ord('g'): # guarda
        bar = np.zeros((img.shape[0],5,3),np.uint8)
        res = np.hstack((img2,bar,img,bar,output))
        cv2.imwrite('salida_grabcut.png',res)
        cv2.imwrite('salida_grabcut1.png',output)
        print " Se ha guardado la imagen\n"
        break
    elif k == ord('r'): # reiniciar todo
        print "reiniciando \n"
        rect = (0,0,1,1)
        drawing = False
        rectangle = False
        rect_or_mask = 100
        rect_over = False
        value = DRAW_FG
        img = img2.copy()
        mask = np.zeros(img.shape[:2],dtype = np.uint8) # reinicia la matriz
        output = np.zeros(img.shape,np.uint8)           # muestrala
    elif k == ord('a'): # haz los segmentos
        print 'Actualizando el corte de la imagen'
        if (rect_or_mask == 0):         # 
            bgdmodel = np.zeros((1,65),np.float64)
            fgdmodel = np.zeros((1,65),np.float64)
            cv2.grabCut(img2,mask,rect,bgdmodel,fgdmodel,1,cv2.GC_INIT_WITH_RECT)
            rect_or_mask = 1
        elif rect_or_mask == 1:        
            bgdmodel = np.zeros((1,65),np.float64)
            fgdmodel = np.zeros((1,65),np.float64)
            cv2.grabCut(img2,mask,rect,bgdmodel,fgdmodel,1,cv2.GC_INIT_WITH_MASK)

    mask2 = np.where((mask==1) + (mask==3),255,0).astype('uint8')
    output = cv2.bitwise_and(img2,img2,mask=mask2)

cv2.destroyAllWindows()
