#!/usr/bin/python

from PIL import Image
import sys, time, random, math

class Filtros:
    
    def __init__(self):
        self.efecto = {
            "grayscale":self.escala_grises,
            "umbral":self.filtro_umbral,
            "media":self.filtro_media,
            "diferencia":self.diferencia_media_grises,
            "generarsl":self.generar_sal_y_pimienta
        }
        
        self.indice_mascara = {
            "gaussian":[[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
            "sobelx":[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            "sobely":[[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]],
            "prewittx":[[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]],
            "prewitty":[[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]]
        }
    
    def aplicar_efecto(self, nImagen, nOutput, aplicar_efectos=[]):
        image = Image.open(nImagen)
        const = 16.0
        for i in aplicar_efectos:
            pic = efecto[i](image)
        image.save(nOutput)
    
    ###################################################################
    # Filtros
    ###################################################################
    def escala_grises(self, image):
        pic = image.load()
        for i in range(image.size[0]):
            for j in range(image.size[1]):
                
                (R, G, B) = pic[i,j]
                # Grayscale
                intensity = int((R+G+B)/3)
                R = G = B = intensity
                pic[i,j] = (R, G, B)
        return pic

    def filtro_media(self, image):
        pic = image.load()
        pic_copy = (image.copy()).load()
        
        for i in range(image.size[0]):
            for j in range(image.size[1]):
                
                temp = [[], [], []]
                for h in range(i-1, i+2):
                    for l in range(j-1, j+2):
                        if h >= 0 and l >= 0 and h < image.size[0] and l < image.size[1]:
                            R, G, B = pic_copy[h, l]
                            temp[0].append(R)
                            temp[1].append(G)
                            temp[2].append(B)
            
                temp[0].sort()
                temp[1].sort()
                temp[2].sort()
                R = int(temp[0][int(len(temp[0])/2)])
                G = int(temp[1][int(len(temp[1])/2)])
                B = int(temp[2][int(len(temp[2])/2)])
                pic[i,j] = (R, G, B)
        return pic

    def diferencia_media_grises(self, image):
        image2 = image.copy()
        self.escala_grises(image)
        self.filtro_media(image2)
        pic = image.load()
        pic2 = image2.load()
        
        # grises - media
        max_values = 0
        for i in range(image.size[0]):
            for j in range(image.size[1]):
                (R, G, B) = pic[i,j]
                (R2, G2, B2) = pic2[i,j]
                R = R2 - R
                G = G2 - G
                B = B2 - B
                if R > max_values:
                    max_values = R
                pic[i,j] = (R, G, B)
        
        pseudo_promedio = self.normalizar(image, [max_values]*3)
        self.filtro_umbral(image, umbral=35)

    def generar_sal_y_pimienta(self, image, densidad=0.15):
        """ Densidad es un valor entre 0 y 1 que  define
            la probabilidad de que se genere ruido en un pixel
            """
        pic = image.load()
        pic_copy = (image.copy()).load()
        for i in range(image.size[0]):
            for j in range(image.size[1]):
                (R, G, B) = pic[i,j]
                if random.random() < densidad:
                    if random.randint(0, 1) == 0:
                        R = G = B = 0
                    else:
                        R = G = B = 255
                pic[i,j] = (R, G, B)
    
    #############################################################
    # Deteccion de bordes con gradiente
    #############################################################
    
    def normalizar(self, image, max_values):
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
    
    def gradient(self, gradientx, gradienty):
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
    
    
    def border_detection(self, picture, output="output.png"):
        imagex = self.aplicar_mascara(picture, ["sobelx"], [1.0/1.0], cmd="i")
        imagey  = self.aplicar_mascara(picture, ["sobely"], [1.0/1.0], cmd="i")
        border, max_values = self.gradient(imagex, imagey)
        del imagey
        
        pseudo_promedio = self.normalizar(border, max_values)
        self.filtro_umbral(border, umbral=80)
        
        border.save(output)
    ##############################################################
    #
    ##############################################################
    def matrix_copy(self, matrix):
        new = list()
        for i in matrix:
            temp = list()
            for j in i:
                temp.append(j)
            new.append(temp)
        return new

    def convolucion(self, kernel, image):
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

    def multiplicar_mascara(self, kernel, const):
        for i in range(len(kernel)):
            for j in range(len(kernel[0])):
                kernel[i][j] *= const
        return kernel

    def aplicar_mascara(self, nImagen, mascara=[], const=[], nOutput="output.jpg", image=None, cmd=""):
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

    def filtro_umbral(self, image, umbral=128):
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


def filtro_promedio(image):
    pic = image.load()
    pic_copy = (image.copy()).load()
    
    for i in range(image.size[0]):
        for j in range(image.size[1]):
            
            temp = [0.0, 0.0, 0.0]
            contador = 0
            for h in range(i-1, i+2):
                for l in range(j-1, j+2):
                    if h >= 0 and l >= 0 and h < image.size[0] and l < image.size[1]:
                        R, G, B = pic_copy[h, l]
                        temp[0] += R
                        temp[1] += G
                        temp[2] += B
                        contador += 1
            pic[i,j] = ( int(temp[0]/contador), int(temp[0]/contador), int(temp[0]/contador))
    return pic

#def main():
#    f = Filtros()
#    f.border_detection(sys.argv[1])   
#main()