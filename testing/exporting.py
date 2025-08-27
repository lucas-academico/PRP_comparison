# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 12:14:15 2025

@author: BANGHO
"""

import pandas as pd
import pyomo.environ as pyo

def export_pyomo_variables_to_excel(m, kind_model='4index', filename="pyomo_variables.xlsx"):
    writer = pd.ExcelWriter(filename, engine='openpyxl')

    # Helper to convert variable to DataFrame
    def var_to_df(var, var_name):
        data = []
        for index in var:
            value = pyo.value(var[index])
            data.append((*index, value))
        df = pd.DataFrame(data, columns=[f"Index_{i+1}" for i in range(len(data[0])-1)] + [var_name])
        return df

    # Export each variable
    if kind_model == '2flow':    
        var_dict = {
            "y": m.y,
            "q": m.q,
            "x": m.x,
            "z": m.z,
            "f": m.f,
            "I": m.I,
            "p": m.p,
            "d": m.d
        }
    else:
        var_dict = {
            "y": m.y,
            "q": m.q,
            "x": m.x,
            "z": m.z,
            "I": m.I,
            "p": m.p,
            "d": m.d
        }

    for var_name, var_obj in var_dict.items():
        df = var_to_df(var_obj, var_name)
        df.to_excel(writer, sheet_name=var_name, index=False)

    writer.close()
    print(f"✅ Variables exported to {filename}")