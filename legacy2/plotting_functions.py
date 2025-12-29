import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math


def plot_histograms(df):
    df_nuevo = df.copy()
    # Definir las columnas que queremos visualizar (excluyendo 'Fecha_Hora' ya que es una fecha)
    # variables = ['Humedad_mean', 'Temperatura_mean', 'Presion_mean', 'Velocidad_mean','Direccion_mean', 'Precipitacion_mean', 'PM25']
    variables = [col for col in df_nuevo.columns if col != "Fecha_Hora"]

    # Crear una figura con subgráficos (ajustar tamaño según cantidad de variables)
    fig, axes = plt.subplots(
        nrows=math.ceil(len(variables) / 3), ncols=3, figsize=(15, 12)
    )  # 4 filas, 3 columnas

    # Aplanar la matriz de ejes para iterar sobre ella fácilmente
    axes = axes.flatten()

    # Generar histogramas para cada variable
    for i, col in enumerate(variables):
        if i < len(variables):  # Evitar errores si hay menos variables que subgráficos
            df_nuevo[col].dropna().astype(float).hist(
                ax=axes[i], bins=30, edgecolor="black", alpha=0.7
            )
            axes[i].set_title(col)
            axes[i].set_xlabel("Valor")
            axes[i].set_ylabel("Frecuencia")

    # Ajustar diseño para evitar solapamientos
    plt.tight_layout()
    plt.show()


def plot_box_and_whiskers(df):
    df_nuevo = df.copy()
    variables = [col for col in df_nuevo.columns if col != "Fecha_Hora"]

    # Tamaño del gráfico
    plt.figure(figsize=(12, 8))

    # Crear un boxplot para cada variable
    for i, col in enumerate(variables, 1):
        plt.subplot(3, math.ceil(len(variables) / 3), i)  # Organiza en una cuadrícula
        sns.boxplot(y=df_nuevo[col])
        plt.title(col)  # Título con el nombre de la variable
        plt.ylabel("Valor")

    plt.tight_layout()  # Ajusta el diseño para evitar sobreposición
    plt.show()
