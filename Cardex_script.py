
"""
Created on Thu Apr 13 22:03:43 2023


@author: 80300698
"""

import pandas as pd 
import xlsxwriter 
import openpyxl 

url = """D:\\users\\80300698\\Pepsico\\Europe Category Finance Team - 2. Databases\\Category Central Folder\\2023\\10. Personal folders\\Dorota\\Cardex\\"""
source_file = "RunTest.xlsx"
tab = "CARDEXS_Total"
split_data_file = "Validacao_Cardex_updated.xlsx"
url_split_data = """D:\\users\\80300698\\Pepsico\\Europe Category Finance Team - 2. Databases\\Category Central Folder\\2023\\10. Personal folders\\Dorota\\Cardex\\Customers\\"""

df = pd.read_excel(url + source_file, tab)

#remove last column & czemu ativo jest w pierwszej kolumnie??

writer = pd.ExcelWriter(url + source_file, engine = 'openpyxl', mode = 'a', if_sheet_exists= 'replace')

entire_table_customers = ('BP', 'Cepsa', 'GALP', 'IBERSOL', 'MEU SUPER', 'Toys'R'Us')

for customer in df['Cardex'].unique():
    newDf = df[df['Cardex'] == customer]
    sheet_name = customer
    newDf_activo = newDf.query("Ativo == 'Sim'")
                               
    if customer in entire_table_customers:
        newDf_activo.to_excel(writer, sheet_name = customer, index=False, startrow=5)
      
    else: 
        newDf_activo.to_excel(writer, sheet_name=sheet_name, index=False, startrow=5, columns=newDf.columns.difference([newDf.columns[3]]))
writer.save()


distributors = {
    'DISNACK': ['Cardex', 'Cod HHC', 'Descrição Adicional'],
    'JAIME ALBERTO': ['Cardex', 'Cod HHC', 'Descrição Adicional'],
    'NORSNACK': ['Cardex', 'Cod HHC', 'Descrição Adicional'],
    'GENIALIS':['Cardex', 'Cod HHC', 'Descrição Adicional'],
    'MADEIRA':['Cardex', 'Cod HHC', 'Descrição Adicional']}

#iterate for every key and every value in dictionary 
for distributor, columns in distributors.items(): 
    df_filtered = df.loc[df['Ativo'] == "Sim", columns]
    df_distributor = df_filtered[columns]
    df_distributor.to_excel(url_split_data + f'Cardex_{distributor}.xlsx', sheet_name = distributor, index = False)

writer.close()

