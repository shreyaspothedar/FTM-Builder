# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 19:51:04 2023

@author: spoth
"""

import numpy as np


class Rechteck:
    def __init__(self, breite, hoehe, negativ=False):
        self.breite = breite
        self.hoehe = hoehe
        self.__schwerpunkt = np.array([self.breite/2, self.hoehe/2])
        self.negativ = negativ

        self.__FTM = (-1)**self.negativ * np.array(
            [[self.breite*(self.hoehe**3)/12 + self.hoehe*(self.breite**3)/12, 0, 0],
             [0, self.breite*(self.hoehe**3)/12, 0],
             [0, 0, self.hoehe*(self.breite**3)/12]
             ])


    def FTM_Tensor(self):
        return self.__FTM

    def setFTM_Tensor(self, new_tensor):
        self.__FTM = new_tensor
        return self.__FTM

    def area(self):
        return (self.breite * self.hoehe) * (-1)**self.negativ

    def getSchwerpunkt(self):
        return self.__schwerpunkt

    def setSchwerpunkt(self, schwerpunkt):
        self.__schwerpunkt = schwerpunkt


class Dreieck:
    def __init__(self, breite, hoehe, negativ=False):
        self.breite = breite
        self.hoehe = hoehe
        self.__schwerpunkt = np.array([self.breite/3, self.hoehe/3])

        self.__FTM = (-1)**self.negativ * np.array(
            [[self.breite*(self.hoehe**3)/36 + self.hoehe*(self.breite**3)/36, 0, 0],
             [0, self.breite*(self.hoehe**3)/36,
              (self.breite**2 * self.hoehe**2)/72],
             [0, (self.breite**2 * self.hoehe**2) /
              72, self.hoehe*(self.breite**3)/36]
             ])
        self.negativ = negativ

    def FTM_Tensor(self):
        return self.__FTM

    def setFTM_Tensor(self, new_tensor):
        self.__FTM = new_tensor
        return self.__FTM

    def area(self):
        return (self.breite * self.hoehe)/2 * (-1)**self.negativ

    def getSchwerpunkt(self):
        return self.__schwerpunkt

    def setSchwerpunkt(self, schwerpunkt):
        self.__schwerpunkt = schwerpunkt
        return self.__schwerpunkt


class Teilkreis:
    def __init__(self, radius, phi, negativ=False):
        self.breite = self.hoehe = self.radius = radius
        self.phi = phi
        self.negativ = negativ

        self.__schwerpunkt = np.array([
            (4*self.radius*(np.sin(self.phi/2)*(np.cos(self.phi/2)))/(3*self.phi)),
            (4*self.radius*(np.sin(self.phi/2))**2)/(3*self.phi)
        ])

        self.__FTM = (-1)**self.negativ * np.array([
            [(self.radius**4)*(self.phi-np.sin(self.phi)*np.cos(self.phi))/8 +
             (self.radius**4)*(self.phi+np.sin(self.phi)*np.cos(self.phi))/8, 0, 0],
            [0, (self.radius**4)*(self.phi-np.sin(self.phi)*np.cos(self.phi)
                                  )/8, (self.radius**4)*((np.sin(self.phi))**2)/8],
            [0, (self.radius**4)*((np.sin(self.phi))**2)/8, (self.radius**4)
             * (self.phi+np.sin(self.phi)*np.cos(self.phi))/8]
        ]) - np.array([
            [self.__schwerpunkt[0]**2 * self.area() + self.__schwerpunkt[1]
             ** 2 * self.area(), 0, 0],
            [0, self.__schwerpunkt[1]**2 * self.area(), self.__schwerpunkt[1] *
             self.__schwerpunkt[0] * self.area()],
            [0, self.__schwerpunkt[1] * self.__schwerpunkt[0] *
                self.area(), self.__schwerpunkt[0]**2 * self.area()]
        ])

    def FTM_Tensor(self):
        return self.__FTM

    def setFTM_Tensor(self, new_tensor):
        self.__FTM = new_tensor
        return self.__FTM

    def area(self):
        return (self.phi*(self.radius**2))/2 * (-1)**self.negativ

    def getSchwerpunkt(self):
        return self.__schwerpunkt

    def setSchwerpunkt(self, schwerpunkt):
        self.__schwerpunkt = schwerpunkt
        return self.__schwerpunkt


#######################

def drehen(obj, winkel):
    J22 = (obj.FTM_Tensor()[1][1] + obj.FTM_Tensor()[2][2])/2 
    + (obj.FTM_Tensor()[1][1] - obj.FTM_Tensor()[2][2]) * np.cos(2*winkel)/2
    + obj.FTM_Tensor()[1][2]*np.sin(2*winkel)
    
    J33 = (obj.FTM_Tensor()[1][1] + obj.FTM_Tensor()[2][2])/2 
    - (obj.FTM_Tensor()[1][1] - obj.FTM_Tensor()[2][2]) * np.cos(2*winkel)/2
    - obj.FTM_Tensor()[1][2]*np.sin(2*winkel)
    
    J23 = - (obj.FTM_Tensor()[1][1] - obj.FTM_Tensor()[2][2]) * np.sin(2*winkel)/2
    + obj.FTM_Tensor()[1][2]*np.cos(2*winkel)
    
    J11 = J22 + J33

    obj.setFTM_Tensor(np.array[[J11, 0, 0], [0, J22, J23], [0, J23, J33]])

    return obj.FTM_Tensor() 


def verschieben(obj, x2, x3):

    obj.setSchwerpunkt(np.array([obj.getSchwerpunkt()[0] + x2,
                       obj.getSchwerpunkt()[1] + x3]))

    return print(f'Der Schwerpunkt des Objektes befindet sich nun bei {obj.getSchwerpunkt().round(4)}.')


def gesamtFlaeche(*args):
    gesamt = 0
    for teil in args:
        gesamt += teil.area()
    return gesamt


def statischesMoment(*args):
    statischesMoment2 = statischesMoment3 = 0
    for elem in args:
        statischesMoment2 += elem.getSchwerpunkt()[0] * elem.area()
        statischesMoment3 += elem.getSchwerpunkt()[1] * elem.area()

    return np.array([statischesMoment2, statischesMoment3])


def gesamtSchwerpunkt(*args):
    return statischesMoment(*args)/gesamtFlaeche(*args)


def spiegeln(obj, koordinate):
    if koordinate.lower() == 'x2':
        obj.setSchwerpunkt(
            np.array([obj.getSchwerpunkt()[0], -obj.getSchwerpunkt()[1]]))
        return obj.getSchwerpunkt()
    elif koordinate.lower() == 'x3':
        obj.setSchwerpunkt(
            np.array([-obj.getSchwerpunkt()[0], obj.getSchwerpunkt()[1]]))
        return obj.getSchwerpunkt()
    else:
        print("Üngültige Eingabe! Antwortmöglichkeiten sind x2 oder x3")


def gesamtFTM(*args):
    steiner = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]).astype(np.float64)
    teilFTM = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]).astype(np.float64)

    gesamtSP = gesamtSchwerpunkt(*args)

    for teile in args:

        s22 = ((gesamtSP[1] - teile.getSchwerpunkt()[1])**2) * teile.area()
        s33 = ((gesamtSP[0] - teile.getSchwerpunkt()[0])**2) * teile.area()
        s23 = ((gesamtSP[0] - teile.getSchwerpunkt()[0]) *
               (gesamtSP[1] - teile.getSchwerpunkt()[1])) * teile.area()

        steiner += np.array([[s22 + s33, 0, 0], [0, s22, s23], [0, s23, s33]])
        teilFTM += teile.FTM_Tensor()

    return (steiner + teilFTM).round(5)


def maxFTM(*args):
    J1 = gesamtFTM(*args)[0][0]/2 - np.sqrt(((gesamtFTM(*args)[1][1] - gesamtFTM(*args)[2][2])**2)/4 + gesamtFTM(*args)[1][2]**2)
    J2 = gesamtFTM(*args)[0][0]/2 + np.sqrt(((gesamtFTM(*args)[1][1] - gesamtFTM(*args)[2][2])**2)/4 + gesamtFTM(*args)[1][2]**2)
    phi0 = np.arctan((2*gesamtFTM(*args)[1][2])/(gesamtFTM(*args)[1][1]-gesamtFTM(*args)[2][2]))/2
    
    return J1, J2, phi0

