import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from IPython.core.display import HTML
from seaborn import distplot
#from scripts_aux import describe_stats as ds
from matplotlib.offsetbox import AnchoredText
#from scripts_aux import prob_calc as pc
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn import metrics
import time


pd.options.display.float_format = '{:,.1f}'.format


# Constructor general


def main_rs_aca_client(df, zip_client, members, zip_camp, df_pov_t, df_pov, ingreso_anual, df_zip, df_rsa_metal,df_hosp):

    df_hosp_a = df_hosp[['NAME','STATE','ADDRESS','ZIP','LATITUDE', 'LONGITUDE','CARRIER']]

    # df = Rsa_aca , df_ref = rmc_aux, zip_camp = 'Zip', df_pov_t = Rsa_pov_t, df_pov = pov_lvl, df_zip = df_zip, df_rsa_metal = Rsa_metal

    state_client = 'FL'

    rank_m = rank_metod(df, zip_client, members)

    rmc_aux = rank_m[['carrier', 'plan', 'Rank_carr_zip']]
    rmc_aux['aux_key'] = rmc_aux['carrier'] + '_' + rmc_aux['plan']
    rmc_aux = rmc_aux[['aux_key', 'Rank_carr_zip']]

    corr_carr = corr_metod(df, rmc_aux, zip_camp, members)

    corr_pov = corr_metod_pov(df, df_pov_t, zip_client,
                              df_pov, ingreso_anual, members, df_zip)

    corr_metal = corr_metod_met(df_rsa_metal, zip_client, df_zip, df)

    Rsa_bdp = pd.concat([rank_m.reset_index(drop=True), corr_carr.reset_index(
        drop=True), corr_pov.reset_index(drop=True), corr_metal.reset_index(drop=True)], axis=0)

    aux_df_1 = df_zip[['Zip', 'Latitude', 'Longitude']]

    # Calculo Cluster y top recomendación

    zip_obj = Rsa_bdp[['Zip', 'Rank_carr_zip']].copy()
    zip_obj = zip_obj.drop_duplicates()

    # DF auxiliar con coordenada

    aux_df_1['Zip'] = aux_df_1['Zip'].astype(int)

    zip_obj_aux = pd.merge(zip_obj, aux_df_1, how='left', on='Zip')

    y_labels = cluster(zip_obj_aux, 5)

    zip_obj_aux['cluster'] = y_labels

    mask_1 = zip_obj_aux['Zip'] == zip_client
    cluster_client = zip_obj_aux[mask_1].copy()

    if cluster_client.empty:
        not_exist = 1
        client_out = zip_obj_aux.copy()
        client_out['Zip'] = client_out['Zip'].astype(int)

    else:
        not_exist = 0
        cluster_client = cluster_client.loc[0, 'cluster']

        mask = zip_obj_aux['cluster'] == cluster_client
        client_out = zip_obj_aux[mask]
        client_out['Zip'] = client_out['Zip'].astype(int)

    # Df inicial resultado del cluster

    client_max_1 = client_out.sort_values('Rank_carr_zip', ascending=False)
    client_max_1 = client_max_1.head(2)

    # Df auxiliar zip code client

    mask = client_out['Zip'] == zip_client
    client_max_aux = client_out[mask].copy()

    if client_max_aux.empty:
        client_max_aux = client_out.copy()

    client_max_aux = client_max_aux.loc[0]

    client_max_aux['Rank_carr_zip'] = max(
        list(client_max_1['Rank_carr_zip'].values))*1.3

    client_max_aux = pd.DataFrame(client_max_aux).T

    client_max_aux['Zip'] = client_max_aux['Zip'].astype(int)

    # Salida al mapa

    client_max = pd.concat([client_max_1.reset_index(
        drop=True), client_max_aux.reset_index(drop=True)], axis=0)

    client_max = client_max.reset_index()

    pivot_client = client_max['Zip'].copy()

    rs_aca_client_out = pd.merge(
        pivot_client, Rsa_bdp, how='left', on='Zip')

    rs_aca_client_out = rs_aca_client_out.drop_duplicates()

    #mask_out = rs_aca_client_out['Total_Members'] == members
    #rs_aca_client_out = rs_aca_client_out[mask_out]

    rs_aca_client_out = rs_aca_client_out.sort_values(
        'Rank_carr_zip', ascending=False)

    # rs_aca_client_out.to_excel('rs_aca_client_out.xlsx')

    # Top carrier

    carrier_top = rs_aca_client_out.groupby('carrier')['plan'].count()
    carrier_top = carrier_top.sort_values(ascending=False)
    carrier_top = carrier_top.head()
    carrier_top = list(carrier_top.index)

    rs_aca_client_out = rs_aca_client_out.head(5)

    rs_aca_client_out = rs_aca_client_out[[
        'carrier', 'plan', 'Total_Members', 'Rank_carr_zip']]

    rs_aca_client_out = rs_aca_client_out.rename(
        columns={'Rank_carr_zip': 'Score Carrier'})

            
    #Top Hospital
     
    best_hosp = hosp(zip_client,df_zip,df_hosp_a,state_client)

    return rs_aca_client_out, carrier_top, not_exist,best_hosp


# Correlación por %Carrier

def corr_metod(df, df_ref, zip_camp, members):

    # metods = ['pearson', 'spearman', 'kendall'] Se deja pearson debido a su nivel de exactitud a ranking grandes
    metods = ['pearson']

    corr_list = []

    df_mask = df['Total_Members'] == members
    df_p = df[df_mask]

    aux_index = df_p.groupby(zip_camp)[zip_camp].nunique()
    aux_index = list(aux_index.index)

    df_aux_out = pd.DataFrame(columns=[
                              'Zip', 'coef_corr', 'carrier', 'plan', 'Total_Members', 'Rank_carr_zip', 'source'])

    aux_comp = df_p[['Zip', 'carrier', 'plan', 'Total_Members',
                     'Rank_carr_zip', 'source']].drop_duplicates()

    for j in range(len(metods)):

        for i in range(len(aux_index)):

            df_mask = aux_comp['Zip'] == aux_index[i]
            aux_comp_1 = aux_comp[df_mask]
            aux_comp_1 = aux_comp_1.rename(
                columns={'Rank_carr_zip': 'Rank_carr_zip_ref'})
            aux_comp_1 = aux_comp_1.drop_duplicates()
            aux_comp_1['aux_key'] = aux_comp_1['carrier'] + \
                '_' + aux_comp_1['plan']

            aux_merge = pd.merge(aux_comp_1, df_ref, how='left', on='aux_key')

            corr_pearson = aux_merge['Rank_carr_zip'].corr(
                aux_merge['Rank_carr_zip_ref'], method=metods[j])

            aux_tup = (aux_index[i], corr_pearson)
            corr_list.append(aux_tup)

        df_pearson = pd.DataFrame.from_records(
            corr_list, columns=['Zip', 'coef_corr'])
        df_pearson = df_pearson.dropna()

        df_mask = df_pearson['coef_corr'] >= 0
        df_pearson = df_pearson[df_mask]

        #df_mask = df_pearson['coef_corr'] >= 0.9
        #df_pearson = df_pearson[df_mask]

        df_pearson = df_pearson.sort_values('coef_corr', ascending=False)
        df_pearson = pd.merge(df_pearson, aux_comp, how='left', on='Zip')

    df_corr_out = pd.concat([df_aux_out.reset_index(
        drop=True), df_pearson.reset_index(drop=True)], axis=0)
    df_corr_out = df_corr_out.sort_values('Rank_carr_zip', ascending=False)
    df_corr_out = df_corr_out.drop_duplicates()
    df_corr_out['Method'] = 'Carr_met'

    return df_corr_out


# Metodo ranking

def rank_metod(df, zip_client, members):

    df_mask = df['Zip'] == zip_client
    df_client = df[df_mask]

    df_mask = df_client['Total_Members'] == members
    df_client = df_client[df_mask]

    df_client = df_client.sort_values('Rank_carr_zip', ascending=False)
    df_client['source_recommend'] = 'ranking methodology'

    df_client = df_client[['Zip', 'carrier', 'plan',
                           'Total_Members', 'Rank_carr_zip', 'source']]
    df_client = df_client.drop_duplicates()

    df_client['Method'] = 'Rank_met'

    return df_client


# Clasificación de nivel de pobreza

def pov_lvl_f(df, inp, members):

    p_lvl = 0

    df_mask = df['family_size'] == members
    aux_df = df[df_mask]

    if inp < aux_df.loc[members-1, 1]:
        p_lvl = '<100%'

    elif inp > aux_df.loc[members-1, 1] and inp <= aux_df.loc[members-1, 1.5]:
        p_lvl = '>100% to <150%'

    elif inp > aux_df.loc[members-1, 1.5] and inp <= aux_df.loc[members-1, 2]:
        p_lvl = '>150% to <200%'

    elif inp > aux_df.loc[members-1, 2] and inp <= aux_df.loc[members-1, 2.5]:
        p_lvl = '>200% to <250%'

    elif inp > aux_df.loc[members-1, 2.5] and inp <= aux_df.loc[members-1, 3]:
        p_lvl = '>250% to <300%'

    elif inp > aux_df.loc[members-1, 3] and inp <= aux_df.loc[members-1, 4]:
        p_lvl = '>300% to <400%'

    elif inp > aux_df.loc[members-1, 4]:
        p_lvl = '>400%'

    return p_lvl


# Metodo corr poverty

def corr_metod_pov(df, df_pov_t, zip_code, df_pov, inp, members, df_zip):

    #"""df = Rsa_aca, df_pov_t = Rsa_pov_t, df_pov = pov_lvl, df_zip = df_zip """

    pov_lvl_c = pov_lvl_f(df_pov, inp, members)

    zip_aux = df_zip[['Zip', 'County']].copy()
    zip_aux['Zip'] = zip_aux['Zip'].astype(int)

    aux_mask_df = zip_aux['Zip'] == zip_code
    aux_df_county = zip_aux[aux_mask_df].copy()
    aux_df_county = aux_df_county['County']
    aux_df_county = list(aux_df_county.values)

    # Alistamiento data
    aux_df = df[['Zip']]

    aux_merge = pd.merge(df_pov_t, aux_df, how='left', on='Zip')
    aux_merge = aux_merge.drop_duplicates()
    aux_merge = aux_merge.dropna()

    df_mask = aux_merge['Clasificacion'] == pov_lvl_c
    rsa_aca_cor = aux_merge[df_mask].copy()

    # Frame auxiliar

    df_mask = rsa_aca_cor['Zip'] == zip_code
    rsa_aca_ref = rsa_aca_cor[df_mask].copy()

    aux_val = list(rsa_aca_ref['Rank_pov_zip'].values)

    # Frame final

    if len(aux_val) == 0:

        aux_mask_1 = rsa_aca_cor['County'] == aux_df_county[0]
        df_aux_corr_client = rsa_aca_cor[aux_mask_1].copy()
        aux_val_aux = list(df_aux_corr_client.groupby(
            'County')['Rank_pov_zip'].median().values)
        rsa_aca_cor['Rank_pov_zip_client'] = aux_val_aux[0]

    else:
        rsa_aca_cor['Rank_pov_zip_client'] = aux_val[0]

    # Calculo de correlaciones

    aux_index = rsa_aca_cor.groupby('County')['County'].nunique()
    aux_index = list(aux_index.index)

    corr_list = []

    # metods = ['pearson', 'spearman', 'kendall'] se debe utilizar solo pearson, debido a que es el que genera muestras significativas.

    metods = ['pearson']

    for j in range(len(metods)):

        for i in range(len(aux_index)):

            df_mask = rsa_aca_cor['County'] == aux_index[i]
            df_aux = rsa_aca_cor[df_mask]

            corr = df_aux['Rank_pov_zip_client'].corr(
                df_aux['Rank_pov_zip'], method=metods[j])

            aux_tup = (aux_index[i], corr)
            corr_list.append(aux_tup)

    df_corr_pov = pd.DataFrame.from_records(
        corr_list, columns=['County', 'coef_corr'])
    df_corr_pov = df_corr_pov.dropna()
    df_corr_pov['coef_corr'] = df_corr_pov['coef_corr'] * \
        1000000000000000  # Corrección de la correlacion

    df_corr_pov = df_corr_pov.sort_values('coef_corr', ascending=False)

    df_aux_rsa = df[['Zip', 'carrier', 'plan',
                     'Total_Members', 'Rank_carr_zip', 'source']]

    df_aux_pov_1 = pd.merge(df_aux_rsa, zip_aux, how='left', on='Zip')

    county_data = df_aux_pov_1['County'].copy()
    county_data = pd.DataFrame(county_data)
    county_data['Ident'] = 'True'

    df_aux_pov = pd.merge(df_corr_pov, county_data, how='left', on='County')

    df_aux_pov = df_aux_pov.dropna()
    df_aux_pov = df_aux_pov.drop_duplicates()

    mask = df_aux_pov['coef_corr'] > 0
    df_aux_pov = df_aux_pov[mask]
    df_aux_pov

    df_out_pov = pd.merge(df_aux_pov_1, df_aux_pov, how='left', on='County')
    df_out_pov = df_out_pov.drop_duplicates()

    mask = df_out_pov['Ident'] == 'True'
    df_out_pov = df_out_pov[mask]

    mask = df_out_pov['Total_Members'] == members
    df_out_pov = df_out_pov[mask]
    df_out_pov = df_out_pov[['Zip', 'coef_corr', 'carrier',
                             'plan', 'Total_Members', 'Rank_carr_zip', 'source']]

    df_out_pov['Method'] = 'Pov_met'

    return df_out_pov


def corr_metod_met(df_rsa_metal, zip_client, df_zip, Rsa_aca):

    # df_rsa_metal = Rsa_metal, zip_client = zip_code_client, df_zip = Zip_codes, Rsa_aca = Rsa_aca / initial data

    Rsa_metal_rs = df_rsa_metal.copy()

    Rsa_metal_rs['Total'] = 0

    for i in list(range(3, 8)):
        Rsa_metal_rs['Total'] = Rsa_metal_rs['Total'] + Rsa_metal_rs.iloc[:, i]

    for j in list(range(3, 8)):

        aux_name = Rsa_metal_rs.iloc[:, j].name
        Rsa_metal_rs["%" +
                     aux_name] = (Rsa_metal_rs.iloc[:, j]/Rsa_metal_rs['Total'])*100

    # Datos cliente

    df_zip_1 = df_zip.copy()
    df_zip_1['Zip'] = df_zip_1['Zip'].astype(int)
    mask_client = df_zip_1['Zip'] == zip_client
    df_zip_1 = df_zip[mask_client].copy()
    df_zip_1 = df_zip_1[['Zip', 'County']]

    # %mayor cantidad de metal level

    aux_count = list(df_zip_1['County'].values)

    mask = Rsa_metal_rs['County'] == aux_count[0]
    df_aux = Rsa_metal_rs[mask].copy()
    df_aux = df_aux[['County', '%Catastrophic',
                     '%Bronze', '%Silver', '%Gold', '%Platinum']]

    df_aux_1 = df_aux.T.reset_index()
    df_aux_1 = df_aux_1.iloc[1:6]
    list_aux = list(df_aux_1.max().values)

    df_corr = Rsa_metal_rs[['County', list_aux[0]]]
    df_corr['rank_ref'] = list_aux[1]

    aux_index = list(df_corr['County'].values)

    # metods = ['pearson', 'spearman', 'kendall'] Se aplica pearson, debido a que es el unico que genera correlaciones
    metods = ['pearson']
    list_corr = []

    for j in range(len(metods)):

        for i in range(len(aux_index)):

            mask = df_corr['County'] == aux_index[i]
            df_aux_corr = df_corr[mask].copy()

            corr = df_aux_corr[list_aux[0]].corr(
                df_aux_corr['rank_ref'], method=metods[j])

            aux_tup = (aux_index[i], corr)
            list_corr.append(aux_tup)

    df_out_1 = pd.DataFrame.from_records(
        list_corr, columns=['County', 'coef_corr'])
    df_out_1 = df_out_1.dropna()
    df_out_1['coef_corr'] = df_out_1['coef_corr']*1000000000000000

    mask = df_out_1['coef_corr'] > 0
    df_out_1 = df_out_1[mask]
    df_out_1 = df_out_1.sort_values('coef_corr', ascending=False)
    df_out_1 = df_out_1.drop_duplicates()

    df_zip_aux = df_zip[['Zip', 'County']].copy()
    df_zip_aux['Zip'] = df_zip_aux['Zip'].astype(int)

    df_aux_mlvl = pd.merge(Rsa_aca, df_zip_aux, how='left', on='Zip')

    df_out_mlvl = pd.merge(df_out_1, df_aux_mlvl, how='left', on='County')

    df_out_mlvl = df_out_mlvl.dropna()

    df_out_mlvl = df_out_mlvl.sort_values('Rank_carr_zip', ascending=False)

    df_out_mlvl = df_out_mlvl[['Zip', 'coef_corr', 'carrier',
                               'plan', 'Total_Members', 'Rank_carr_zip', 'source']]

    df_out_mlvl['Zip'] = df_out_mlvl['Zip'].astype(int)

    df_out_mlvl = df_out_mlvl.drop_duplicates()

    df_out_mlvl['Method'] = 'Metal_met'

    return df_out_mlvl


def n_k(df):

    Nc = range(1, 20)
    kmeans = [KMeans(n_clusters=i) for i in Nc]
    kmeans
    score = [kmeans[i].fit(df).score(df)
             for i in range(len(kmeans))]
    score
    plt.plot(Nc, score)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Score')
    plt.title('Elbow Curve')
    plt.show()


def cluster(df, n_cluster):

    model = KMeans(n_clusters=n_cluster, max_iter=1000)
    model.fit(df)
    y_labels = model.labels_

    y_kmeans = model.predict(df)

    return y_labels


# Hospitales mas cercanos

def hosp(zip_client,df_zip,df_hosp_a,state_client):

    df_county = df_zip[['Zip', 'County']].copy()
    df_county['Zip'] = df_county['Zip'].astype(int)
    df_county = df_county.rename(columns={'Zip': 'ZIP'})

    mask_zip = df_county['ZIP'] == zip_client
    county_client = df_county[mask_zip]
    county_client = list(county_client['County'].values)[0]


    df_hosp_c = pd.merge(df_hosp_a, df_county, how='left', on='ZIP')

    mask_hosp = df_hosp_c['STATE'] == state_client
    df_hosp_c = df_hosp_c[mask_hosp]

    hosp_cluster = df_hosp_c[['ZIP', 'LATITUDE', 'LONGITUDE']]

    df_hosp_c['Cluster'] = list(cluster(hosp_cluster, 6))

    # Hospitales en el mismo zip code

    mask_zip = df_hosp_c['ZIP'] == zip_client
    hosp_r_1 = df_hosp_c[mask_zip].copy()

    # Hospitales en el mismo county level

    mask_county = df_hosp_c['County'] == county_client
    hosp_r_2 = df_hosp_c[mask_zip].copy()

    hosp_r = pd.concat([hosp_r_1.reset_index(drop=True),
                        hosp_r_2.reset_index(drop=True)], axis=0)
    hosp_r = hosp_r.drop_duplicates()
    hosp_r = hosp_r[['NAME', 'STATE', 'County', 'ZIP', 'ADDRESS']]

    return hosp_r

    

def eucli(pivot,obje_data): ## función final para calculo de ranking y clasificación


    a = pivot.values

    len_1 = obje_data.shape[0]
    dist_list = []

    for i in range(len_1):

        b = obje_data.iloc[i,1:].values
        dist = np.linalg.norm(a-b)
        dist_list.append(dist)

    corr_2 = list()
    corr_2 = (1 - (dist_list/max(dist_list)))*5
    clasi = []

    for j in range(len(corr_2)):

        if corr_2[j] >= 4:
            clasi.append('Rank 4')

        elif corr_2[j] < 4 and corr_2[j]  >= 3:
            clasi.append('Rank 3')

        elif corr_2[j] < 3 and corr_2[j]  > 2:
            clasi.append('Rank 2')

        elif corr_2[j] <= 2: 
            clasi.append('Rank 1')

        elif corr_2[j] <= 0:
            clasi.append('Correlación negativa')

        else:        
            clasi.append('Sin clasificación')

    return corr_2, clasi, dist_list