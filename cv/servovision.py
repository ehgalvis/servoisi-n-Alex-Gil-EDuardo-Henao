import cv2
import numpy as np
from numpy import ones, vstack
from numpy.linalg import lstsq
from sklearn import linear_model
import serial
import time
estado = 0

#comunicación
ser = serial.Serial('/dev/ttyACM0', 115200)
print(ser.readline())


capture = cv2.VideoCapture(0)
#capture = cv2.VideoCapture('uno.mp4')
first = True



#while(capture.isOpened()):
while (True):



    _, frame = capture.read()
    rows,cols, _ = frame.shape

    M = cv2.getRotationMatrix2D((cols/2,rows/2),0,1)

    frame = cv2.warpAffine(frame,M,(cols,rows))


    # frame = frame[0:1700, 0:1000]     #Se RECORTA IMAGEN

    # Se convierte a HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


    lower = np.array([43, 45, 71])
    upper = np.array([77, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)


    kernel = np.ones((5, 5), np.uint8)
    erode = cv2.erode(mask, kernel, iterations=3)
    dilate = cv2.dilate(erode, kernel, iterations=1)


    # Se hace una copia para mostrar al usuario
    dilate_cp = dilate.copy()


    # Se identifica donde hay pixeles blancos
    index = np.where(dilate == [255])

    X = index[0]
    Y = index[1]

    Y = np.transpose([Y])
    X = np.transpose([X])

    if(len(X)>10):


        # Metodo tomado de http://scikit-learn.org/stable/auto_examples/linear_model/plot_ransac.html

        # Ransac 1
        ransac = linear_model.RANSACRegressor()

        ransac.fit(X, Y)

        inliers = ransac.inlier_mask_
        outliers = np.logical_not(inliers)

        # Predict data of estimated models
        line_X = np.arange(X.min(), X.max())[:, np.newaxis]

        line_Y = ransac.predict(line_X)


        # Se borran los puntos
        dilate[X[inliers],Y[inliers]] = 0






        # Se vuelve a hacer ransac para la segunda linea

        index = np.where(dilate == [255])

        X = index[0]
        Y = index[1]

        if(len(X)>10):

            Y = np.transpose([Y])
            X = np.transpose([X])

            ransac = linear_model.RANSACRegressor()

            ransac.fit(X, Y)

            inliers = ransac.inlier_mask_
            outliers = np.logical_not(inliers)

            # Predict data of estimated models
            line_X2 = np.arange(X.min(), X.max())[:, np.newaxis]

            line_Y2 = ransac.predict(line_X2)

            # Se borran el resto de puntos
            dilate[X[inliers],Y[inliers]] = 0





            # Se saca la primera ecuacion

            points = [(line_X[1][0], line_Y[1][0]), (line_X[2][0], line_Y[2][0])]
            p_X, p_Y = zip(*points)
            A0 = vstack([p_X, ones(len(p_X))]).T
            m, c = lstsq(A0, p_Y, rcond=None)[0]



            # Se saca la segunda ecuación

            points2 = [(line_X2[1][0], line_Y2[1][0]), (line_X2[2][0], line_Y2[2][0])]
            p_X2, p_Y2 = zip(*points2)
            A2 = vstack([p_X2, ones(len(p_X2))]).T
            m2, c2 = lstsq(A2, p_Y2, rcond=None)[0]


            # Formula para interescción de las lineas

            x = ((c * m2) - (c2 * m)) / (m2 - m)
            y = (x - c2) / m2






            # Se calcula que tan desviadas están las lineas del centro

            centro = (cols / 2)
            delta_imagen = x - centro
            print(delta_imagen)


            # Se envia a arduino via string / Se  usa MEF que cambia de estado izq - der
            if (estado == 0):

                # No gire el motor
                print("recto")
                msg = 'q'
                ser.write(msg.encode("utf-8"))
                print(msg)


                if (delta_imagen > 10):
                    estado = 1

                elif (delta_imagen < -10):
                    estado = 2

                else:
                    estado = 3


            elif (estado == 1):

                # gire a la izquierda
                msg = 'i'
                print("iiiiiiiiiiiiiiiiiiiiiiiiii")
                ser.write(msg.encode("utf-8"))
                print(msg)

                if (delta_imagen < -10):
                    estado = 2

                elif ((delta_imagen >= -10) | (delta_imagen <= 10)):
                    estado = 0

                else:
                    estado = 1



            elif (estado == 2):

                # gire a la derecha
                msg = 'd'
                print("deeeeeeeeeeeeeeeeeeeeeeeee")
                ser.write(msg.encode("utf-8"))
                print(msg)

                if (delta_imagen > 10):
                    estado = 1

                elif ((delta_imagen >= -10) | (delta_imagen <= 10)):
                    estado = 0

                else:
                   estado = 2


            elif (estado == 3):

                # perdido
                msg = 'q'
                print("perdido")
                ser.write(msg.encode("utf-8"))
                print(msg)

                if (delta_imagen > 10):
                    estado = 1

                elif (delta_imagen < -10):
                    estado = 2

                elif ((delta_imagen >= -10) | (delta_imagen <= 10)):
                    estado = 0


            else:
                # buscar linea
                print("buscar linea")
                msg = 'i'
                ser.write(msg.encode("utf-8"))
                print(msg)

        else:
            #No gire el motor
            print("buscando linea")
            msg = 'd'
            ser.write(msg.encode("utf-8"))
            print(msg)

    else:
        #No gire el motor
        print("no se encuentra ninguna linea")
        msg = 'q'
        ser.write(msg.encode("utf-8"))

    cv2.imshow("lineas", dilate_cp)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
