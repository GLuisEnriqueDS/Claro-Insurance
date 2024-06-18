from seaborn import distplot
from IPython.core.display import display, HTML
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText


def imprimir():

    print('oki doki')

def stat(df): #Genera todo en análisis estadistico

    text_1 = 'Descripción estadistica de las variables'
    print(text_1.center(100, "=") + "\n")
    
    text_2 = 'Descripción Variables Numericas'
    print(text_2.title() + "\n")

    aux_1 = df.describe(include=[np.number])  # describe number type
    display(HTML(aux_1.to_html()))

    text_3 = 'Descripción Variables Discretas'
    print("\n" + text_3.title() + "\n")

    aux_2 = df.describe(include=[object])  # describe object type
    display(HTML(aux_2.to_html()))

    text_4 = 'Los datos a analizar pueden llegar a tener valores atipicos debido a que su valor maximo se aleja demasiado de la media.\n' + \
        'Por lo cual se proceden a calcular algunas estadisticas descriptivas a adicionales, dicho proceso se realiza para cada una de las variables:'

    print("\n" + text_4 + "\n")

    aux = list(aux_1.columns)

    for i in range(len(aux)):  # Calcula medianas y compara

        media = df[aux[i]].mean()
        mediana = df[aux[i]].median()
        #moda = df[aux[i]].mode()

        text = "Para la variable " + aux[i] + " se obtiene" + "\n" + "\nmedia = " + str(
            media) + "\nmediana = " + str(mediana)  # + "\nmoda = " + str(moda)
        print(text + "\n " + "\nDe lo anterior se determina que la mejor medida te tendencia central es la mediana." + "\n "*2)

    text_5 = 'Validación de distribuciones estadísticas'
    print("\n"*2 + text_5.center(100, "=") + "\n"*2)

    plt.rcParams["figure.figsize"] = (10, 4)

    for i in range(len(aux)):  # Grafica los histogramas de cada variable numerica
        print("\n" + "Histograma " + aux[i] + "\n")
        sns.histplot(df[aux[i]], color='#F1C40F',
                     label="100% Equities", kde=True, linewidth=0)
        plt.show()

    text_6 = 'Validación de correlaciones'
    print("\n"*2 + text_6.center(100, "=") + "\n"*2)

    text_7 = 'Comportamiento variables'
    print(text_7.title() + "\n")

    aux_n = df[aux]
    plt.rcParams["figure.figsize"] = (15, 6)
    aux_n.plot()  # Grafica el comportamiento de las variables
    plt.show()

    text_8 = 'Mapa de correlaciones'
    print("\n" + text_8.title() + "\n")

    sns.heatmap(aux_n.corr(), annot=True, fmt='.2f',
                cmap="YlGnBu")  # Mapa de calor correlaciones
    plt.show()

    text_9 = 'Validación de dispersión'
    print("\n"*2 + text_9.center(100, "=") + "\n"*2)

    plt.rcParams["figure.figsize"] = (10, 4)

    for i in range(len(aux)):  # Grafica los histogramas de cada variable numerica
        print("\n" + "Box plot " + aux[i] + "\n")
        sns.boxplot(x=df[aux[i]])
        plt.show()

    text_10 = "Se determina que cada una de las variables mas relevantes en el analisis presentan un cantidad considerable de valores atipicos, \nsin embargo no se realizara" \
        + "ninguna accion de momento sobre estos debido a su impacto en el comportamiento de las variables."
    print("\n" + text_10 + "\n")


def pie(df,ax,ay,exp,explode,title,pos_label='lower right'): #genera un grafico tipo pie a partir de una serie
    
    if exp == 'yes':
        explode = explode
    else:
        explode = [0]*len(list(df.index))  

    plt.rcParams["figure.figsize"] = (ax, ay)
    
    plt.pie(df, autopct='%1.1f%%',explode=explode,
            shadow=True, startangle=0, labels=df.values)

    plt.legend(loc=pos_label, title="LOB", labels=df.index)
    plt.title(title, size=16)

    # Texto

    plt.text(-0.7, -1,  "Total: " + df.sum().astype(str),
            fontsize=13,
            color="black",
            verticalalignment='top',
            horizontalalignment='center',
            bbox={'facecolor': 'white',
                'pad': 10}
            )


def bar_v(df,ax,ay,title,x_label,y_label,min_y,max_y):
    
    plt.rcParams["figure.figsize"] = (ax,ay)


    ax = df.plot(kind='bar')
    plt.title(title, size = 16)
    plt.xlabel(x_label, size = 14)
    plt.ylabel(y_label, size = 14)
    ax.set_ylim([min_y, max_y])
    ax.set_xticklabels(df.index, rotation = 90)


    def add_value_labels(ax, spacing=5):

        # For each bar: Place a label
        for rect in ax.patches:
            # Get X and Y placement of label from rect.
            y_value = rect.get_height()
            x_value = rect.get_x() + rect.get_width() / 2

            # Number of points between bar and label. Change to your liking.
            space = spacing
            # Vertical alignment for positive values
            va = 'bottom'

            # If value of bar is negative: Place label below bar
            if y_value < 0:
                # Invert space to place label below
                space *= -1
                # Vertically align label at top
                va = 'top'

            # Use Y value as label and format number with one decimal place
            label = "{:.1f}".format(y_value)

            # Create annotation
            ax.annotate(
                label,                      # Use `label` as label
                (x_value, y_value),         # Place label at end of the bar
                xytext=(0, space),          # Vertically shift label by `space`
                textcoords="offset points", # Interpret `xytext` as offset in points
                ha='center',                # Horizontally center label
                va=va)                      # Vertically align label differently for
                                    # positive and negative values.

    # Call the function above. All the magic happens there.
    add_value_labels(ax)

def top(df,ax,ay,title,x_label,y_label,box_tex,xmin,xmax):
    
    fig = plt.figure(dpi=100, figsize=(ax, ay))

    ax = fig.gca()
    ax.set_xlim([df.min()-xmin, df.max()+xmax])
    sns.barplot(x=list(df.values), y=list(df.index), ax=ax, orient="h")
    ax.set_title(title)
    ax.set_ylabel(x_label)
    ax.set_xlabel(y_label)

    # Fuente
    text = AnchoredText(box_tex, loc=4,
                        prop={'size': 10, }, frameon=True)
    ax.add_artist(text)


    # Datos barras

    rects = ax.patches

    # For each bar: Place a label
    for rect in rects:
        # Get X and Y placement of label from rect.
        x_value = rect.get_width()
        y_value = rect.get_y() + rect.get_height() / 2

        # Number of points between bar and label. Change to your liking.
        space = 5
        # Vertical alignment for positive values
        ha = 'left'

        # If value of bar is negative: Place label left of bar
        if x_value < 0:
            # Invert space to place label to the left
            space *= -1
            # Horizontally align label at right
            ha = 'right'

        # Use X value as label and format number with one decimal place
        label = "{:.1f}".format(x_value)

        # Create annotation
        plt.annotate(
            label,                      # Use `label` as label
            (x_value, y_value),         # Place label at end of the bar
            xytext=(space, 0),          # Horizontally shift label by `space`
            textcoords="offset points",  # Interpret `xytext` as offset in points
            va='center',                # Vertically center label
            ha=ha)                      # Horizontally align label differently for
        # positive and negative values.


def rango(df,column_clasi,col_out):
    
    # Calculo cuartiles para rangos


    aux = []

    for i in range(5):
        x = 0.25*i
        a = df[column_clasi].quantile(x)
        aux.append(a)

    # Creación Columna vacia

    df[col_out] = ""

    # Clasificación

    for i in range(len(df[column_clasi])):
        try:
            if pd.isnull(df.loc[i, column_clasi]) == True:
                df.loc[i, col_out] = 'Unknown'

            elif df.loc[i, 'premium'] == '':
                df.loc[i, col_out] = 'Unknown'

            elif (df.loc[i, column_clasi] > aux[0] and df.loc[i, column_clasi] < aux[1]):
                df.loc[i, col_out] = 'Rango 1 (' + str(aux[0]) + " & " + str(aux[1]) + ")"
                #premium_range.append('Range 1')

            elif df.loc[i, column_clasi] >= aux[1] and df.loc[i, column_clasi] < aux[2]:
                df.loc[i, col_out] = 'Rango 2 (' + str(aux[1]) + " & " + str(aux[2]) + ")"
                #premium_range.append('Range 2')

            elif df.loc[i, column_clasi] > aux[2] and df.loc[i, column_clasi] < aux[3]:
                df.loc[i, col_out] = 'Rango 3 (' + str(aux[2]) + " & " + str(aux[3]) + ")"
                #premium_range.append('Range 3')

            elif df.loc[i, column_clasi] >= aux[3] and df.loc[i, column_clasi] < aux[4]:
                df.loc[i, col_out] = 'Rango 4 (' + str(aux[3]) + " & " + str(aux[4]) + ")"
                #premium_range.append('Range 4')

            elif df.loc[i, column_clasi] > aux[4]:
                df.loc[i, col_out] = 'Rango 5 > ' + str(aux[4])
                #premium_range.append('Range 5')

            elif (df.loc[i, column_clasi] == 0):
                df.loc[i, col_out] = 'Zero'

        except Exception:
            pass

    # Agrupación y ajuste

    aux_1 = df.groupby(col_out)[col_out].count()
    aux_1 = aux_1.rename({'': 'Not Determined'}).sort_values(ascending=False)

    return aux_1



def life_client(df,age,members,status_name,status_1,status_2,status_3,plan):
    
    # Rango de edades


    mask = (df[age] >= 30) & (df[age] <= 50)
    df = df.loc[mask]


    # Hijos o dependientes

    df_mask = df[members] > 1
    df = df[df_mask]

    # Exclusión Late y Binder Payment

    mask = (df[status_name] != status_1) & (
        df[status_name] != status_2)
    df = df.loc[mask]

    mask = df[status_name] != status_3
    df = df.loc[mask]

    # Niveles de poliza

    df[plan] = df[plan].astype(str)  # Adaptación datos

    silver = df[df.plan.str.contains('Silver')]
    gold = df[df.plan.str.contains('Gold')]
    df = gold.append(silver)

    # Prob de que la persona sea mayor al 150% de nivel de pobreza mayor al 72%

    #client_life = pd.merge(client_life, prob_tot, how='left', on='state_name')

    #df_mask = client_life['Prob_mayor_150'] > 0.7
    #client_life = client_life[df_mask]

    # Calculo mensualidad

    #client_life['monthly_premium'] = client_life['Final_Premium__c'] - \
        #client_life['Subsidy__c']

    return df


def name_column(columns):
    
    name = []

    for i in range(len(columns)):
        aux = columns[i]
        aux = aux[3]
        name.append(aux)
        
    return name
