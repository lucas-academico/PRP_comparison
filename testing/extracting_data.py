# -*- coding: utf-8 -*-
"""
Created on Mon May 26 13:31:16 2025

@author: BANGHO
"""

import pandas as pd

def read_data(size, problem, instance):
    
    data_supplier = []
    data_retailers = []
    link = "DATI_" + str(size) + "\ABS" + str(problem) + "_" + str(size) + "_" +str(instance) + ".dat"    
    
    with open(link, "r") as f:
        leer_datos_sup = False  # Control para identificar la sección de datos
        leer_datos_ret = False
        for line in f:
            line = line.strip()
            if "VEHICLE CAPACITY" in line:
                cap_veh = int(line.split()[2])
            if "SUPPLIER_COORD" in line:
                leer_datos_sup = True
                continue
            if "RETAILER_COORD" in line:
                leer_datos_sup = False
                leer_datos_ret = True
                continue
            if leer_datos_sup and line:
                valores = line.split()
                data_supplier.append(valores)
            
            if leer_datos_ret and line:  
                valores = line.split()  
                data_retailers.append(valores)
    
    # Convertir a DataFrame
    columnas_sup = ["ID", "X", "Y", "I_INIT", "h", "PRODUCTION_COST", "SET_UP_COST"]
    columnas_ret = ["ID", "X", "Y", "I_INIT", "I_MAX", "DROP_THIS_COL", "DEM", "h"]
    df_sup = pd.DataFrame(data_supplier, columns=columnas_sup)
    df_ret = pd.DataFrame(data_retailers, columns=columnas_ret)
    
    return cap_veh, df_sup, df_ret