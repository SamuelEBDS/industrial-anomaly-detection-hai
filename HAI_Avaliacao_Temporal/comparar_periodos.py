"""Avaliação temporal HAI: treina no passado, calibra no passado e testa sem retocar o limiar."""
from __future__ import annotations
import argparse
import json
import platform
import warnings
from hashlib import file_digest
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, f1_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

MODELOS = ['RF com pesos', 'IF sem PCA', 'IF com PCA', 'Regressão Logística', 'Ensemble OU']
PLANO = {
    'J1': ([1], 2, [('21.03', 3)]),
    'J2': ([1, 2], 3, [('21.03', 4)]),
    'J3': ([1, 2, 3], 4, [('21.03', 5)]),
    'EXT': ([1, 2, 3], 4, [('21.03', 5)] + [('22.04', k) for k in range(1, 5)]),
}
SEED = 42
def nomes(ver, k):
    return f"test{k}.csv" + (".gz" if ver == "21.03" else "")

def garantir(ok, msg):
    if not ok:
        raise ValueError(msg)

def metadados(ver, k, df):
    a = df.time.iloc[0]; b = df.time.iloc[-1]
    return {'versao': ver, 'arquivo': f'test{k}', 'inicio': str(a),'fim': str(b),
            'registros': len(df), 'normais': int((df.attack==0).sum()),
            'registros_ataque': int(df.attack.sum())}

def ler(p, ver, k, cols=None):
    rawtime,rawy = ('time','attack') if ver=='21.03' else ('timestamp','Attack')
    df = pd.read_csv(p, usecols=([rawtime,rawy,*cols] if cols is not None else None),
                     parse_dates=[rawtime],low_memory=False).rename(columns={rawtime:'time',rawy:'attack'})
    garantir(df.columns.is_unique and df.time.notna().all() and df.time.is_monotonic_increasing and df.time.is_unique,
             f'Horários/colunas inválidos: {p}')
    garantir(df.attack.isin([0,1]).all() and df.attack.nunique()==2, f'Rótulos inválidos: {p}')
    return df

def industriais(cols):
    return [c for c in cols if c.lower() not in {'time','timestamp','attack','arquivo_origem','indice_origem'}
            and not c.lower().startswith('attack_')]

def X(df, cols):
    a = np.ascontiguousarray(df.loc[:,cols].to_numpy(dtype=np.float64))
    garantir(np.isfinite(a).all(), 'Sinais ausentes ou infinitos')
    return a

def corte(y,s):
    p,r,t = precision_recall_curve(y,s,pos_label=1)
    p,r=p[:-1],r[:-1]
    f=np.divide(2*p*r,p+r,out=np.zeros_like(p),where=p+r!=0)
    return float(t[np.flatnonzero(f==f.max())[-1]])

def episodio_e_metricas(df, alarmes, modelo, rodada, versao, k):
    y=df.attack.to_numpy(dtype=bool); t=df.time.to_numpy(dtype='datetime64[s]')
    primeiro=np.r_[True, np.diff(t)!=np.timedelta64(1,'s')]
    ini=np.flatnonzero(y & (primeiro | ~np.r_[False,y[:-1]]))
    fim=np.flatnonzero(y & (np.r_[primeiro[1:],True] | ~np.r_[y[1:],False]))
    garantir(len(ini)==len(fim) and int((fim-ini+1).sum())==int(y.sum()),'Contagem de episódios')
    eps=[]; atrasos=[]
    for i,(a,b) in enumerate(zip(ini,fim),1):
        hits=np.flatnonzero(alarmes[a:b+1]); atraso=int((t[a+hits[0]]-t[a])/np.timedelta64(1,'s')) if len(hits) else None
        if atraso is not None: atrasos.append(atraso)
        eps.append(dict(rodada=rodada,versao=versao,arquivo=f'test{k}',episodio=i,modelo=modelo,
                        inicio=str(t[a]),fim=str(t[b]),registros=int(b-a+1),sinalizados=len(hits),
                        atraso_segundos=atraso))
    tn, fp, fn, tp=map(int,confusion_matrix(y,alarmes,labels=[False,True]).ravel())
    stats=dict(rodada=rodada,versao=versao,arquivo=f'test{k}',modelo=modelo,registros=len(y),normais=tn+fp,
               registros_ataque=tp+fn,VP=tp,FP=fp,FN=fn,VN=tn,
               precisao=tp/(tp+fp) if tp+fp else 0.,recall=tp/(tp+fn),
               f1=2*tp/(2*tp+fp+fn),taxa_fp=fp/(fp+tn),episodios_total=len(eps),
               episodios_sinalizados=len(atrasos),
               atraso_mediano_s=float(np.median(atrasos)) if atrasos else None)
    garantir(abs(stats['f1']-f1_score(y,alarmes))<1e-12,'F1')
    return eps,stats

def rodada(args,code,ref):
    base21,base22=args.hai21,args.hai22
    root=args.saida; root.mkdir(parents=True,exist_ok=True)
    ids,v,alvos=PLANO[code]; start=perf_counter()
    def path(ver,k):
        return (base21 if ver=='21.03' else base22)/nomes(ver,k)
    def checkfile(ver,k):
        p=path(ver,k)
        with p.open('rb') as fd:
            garantir(file_digest(fd,'sha256').hexdigest()==ref['arquivos'][f'hai-{ver}/{p.name}'], f'Hash diferente: {p}')
    for ver,k in sorted(set([('21.03',i) for i in ids+[v]]+alvos)):
        checkfile(ver,k)
    partes=[ler(path('21.03',i),'21.03',i) for i in ids]
    train=pd.concat(partes,ignore_index=True)
    garantir(train.time.is_unique and train.time.is_monotonic_increasing, 'Treino fora de ordem')
    dados_treino=[metadados('21.03',i,d) for i,d in zip(ids,partes)];del partes
    cand=industriais(train.columns)
    const=[c for c in cand if train[c].nunique(dropna=False)<=1]
    cols=[c for c in cand if c not in const]; excluidas=[]
    if code=='EXT':
        # Só se consultam os nomes das colunas externas para compatibilidade, nunca seus valores/rótulos.
        headers=set.intersection(*(set(pd.read_csv(path('22.04',i),nrows=0).columns) for i in range(1,5)))
        excluidas=[c for c in cols if c not in headers];cols=[c for c in cols if c in headers]
    val=ler(path('21.03',v),'21.03',v,cols)
    garantir(train.time.iloc[-1]<val.time.iloc[0], 'Sobreposição treino/validação')
    if code=='J3': garantir(len(cols)==60,'Protocolo principal: esperados 60 sinais.')
    if code=='EXT': garantir(len(cols)==52,'Protocolo externo: esperados 52 sinais.')
    a=X(train,cols); y=train.attack.to_numpy(dtype=np.int8); del train
    av=X(val,cols); yv=val.attack.to_numpy(dtype=np.int8)
    rf=RandomForestClassifier(n_estimators=100,class_weight='balanced_subsample',random_state=SEED,n_jobs=args.jobs)
    lr=make_pipeline(StandardScaler(),LogisticRegression(solver='lbfgs',C=1.,l1_ratio=0.,class_weight=None,max_iter=1000,tol=1e-6))
    kw=dict(n_estimators=100,max_samples=256,contamination='auto',random_state=SEED,n_jobs=args.jobs)
    i0,i1=IsolationForest(**kw),IsolationForest(**kw); ss=StandardScaler(); pca=PCA(n_components=.95,svd_solver='full')
    print(f'{code}: treino {ids}, validação {v}, {len(cols)} sinais, {len(a)} registros',flush=True)
    rf.fit(a,y)
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        lr.fit(a,y)
    z=ss.fit_transform(a[y==0]); i0.fit(z);zp=pca.fit_transform(z);i1.fit(zp)
    del a,y,z,zp
    def scores(a):
        n=ss.transform(a)
        return [rf.predict_proba(a)[:,1],-i0.score_samples(n),-i1.score_samples(pca.transform(n)),lr.predict_proba(a)[:,1]]
    val_s=scores(av); cut={m:corte(yv,s) for m,s in zip(MODELOS[:4],val_s)}
    def aplicar(s):
        p=[score>=cut[m] for m,score in zip(MODELOS[:4],s)]
        return p+[p[3]|p[2]]
    val_f1={m:float(f1_score(yv,p)) for m,p in zip(MODELOS,aplicar(val_s))}
    versoes={x:version(x) for x in ['numpy','pandas','scipy','scikit-learn','joblib','threadpoolctl']}
    conf=dict(rodada=code,treino=dados_treino,validacao=metadados('21.03',v,val),sinais=cols,
              constantes_so_no_treino=const,ausentes_na_versao_externa=excluidas,
              pca_componentes=int(pca.n_components_),variancia_retida=float(pca.explained_variance_ratio_.sum()),
              limiares=cut,regra_ensemble='OU dos alertas da Regressão Logística e IF com PCA',
              f1_validacao=val_f1,semente_modelos=SEED,versoes=versoes,python=platform.python_version())
    (root/f'protocolo_{code}.json').write_text(json.dumps(conf,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(f'{code}: cortes fixados; começando a avaliação (PCA: {pca.n_components_} componentes).',flush=True)
    all_ep=[];all_m=[];period=[];fim_val=val.time.iloc[-1];del val,av,val_s
    for versao,k in alvos:
        d=ler(path(versao,k),versao,k,cols)
        garantir(fim_val<d.time.iloc[0],f'Teste {versao}/test{k} deve ser posterior à validação')
        if period: garantir(pd.Timestamp(period[-1]['fim'])<d.time.iloc[0],'Testes devem ser disjuntos')
        period.append(metadados(versao,k,d))
        preds=aplicar(scores(X(d,cols)))
        for m,pred in zip(MODELOS,preds):
            ep,met=episodio_e_metricas(d,pred,m,code,versao,k);all_ep+=ep;all_m.append(met)
        print(f'{code}: {versao}/test{k}, {len(d)} linhas avaliadas.',flush=True)
    pd.DataFrame(all_m).to_csv(root/f'metricas_{code}.csv',index=False,float_format='%.12g')
    pd.DataFrame(all_ep).to_csv(root/f'episodios_{code}.csv',index=False)
    pd.DataFrame(period).to_csv(root/f'periodos_{code}.csv',index=False)
    print(f'{code}: concluída em {perf_counter()-start:.1f} s.',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--hai21',type=Path,required=True,help='Pasta com test1.csv.gz ... test5.csv.gz do HAI 21.03')
    ap.add_argument('--hai22',type=Path,required=True,help='Pasta com test1.csv ... test4.csv do HAI 22.04')
    ap.add_argument('--saida',type=Path,default=Path('resultados'));ap.add_argument('--jobs',type=int,default=3)
    ap.add_argument('--rodadas',nargs='+',choices=list(PLANO),default=list(PLANO))
    args=ap.parse_args();ref=json.loads(Path(__file__).with_name('fontes.json').read_text(encoding='utf-8'))
    with threadpool_limits(limits=2,user_api='blas'):
        for code in args.rodadas: rodada(args,code,ref)
