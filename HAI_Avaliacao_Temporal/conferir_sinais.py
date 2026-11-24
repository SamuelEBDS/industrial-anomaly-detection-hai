"""Diagnóstico descritivo PÓS-teste. Não altera entradas, modelos ou limiares."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from comparar_periodos import ler, nomes, X

def conferir(d21,d22,saida):
    feat=json.loads((saida/'protocolo_EXT.json').read_text(encoding='utf-8'))['sinais']
    def normais(ver, ids):
        root=d21 if ver=='21.03' else d22
        groups=[]
        for k in ids:
            f=ler(root/nomes(ver,k),ver,k,feat)
            groups.append(X(f.loc[f.attack.eq(0)],feat))
        return np.concatenate(groups)
    train=normais('21.03',[1,2,3]); low=train.min(axis=0);hi=train.max(axis=0);central=np.median(train,axis=0);del train
    rows=[]; brief=[]
    for label,ver,ids in [('histórico_21.03','21.03',[5]),('externo_22.04','22.04',[1,2,3,4])]:
        a=normais(ver,ids);outside=(a<low)|(a>hi);med=np.median(a,axis=0)
        for j,featname in enumerate(feat):
            rows.append(dict(periodo=label,sinal=featname,min_treino_normal=low[j],max_treino_normal=hi[j],
                mediana_treino_normal=central[j],mediana_teste_normal=med[j],normais=len(a),
                fora_do_intervalo=int(outside[:,j].sum()),fracao_fora=float(outside[:,j].mean()),
                mediana_fora_do_intervalo=bool(med[j]<low[j] or med[j]>hi[j])))
        brief.append(dict(periodo=label,sinais=len(feat),normais=len(a),
                          sinais_com_mediana_fora=int(((med<low)|(med>hi)).sum()),
                          registros_com_algum_sinal_fora=int(outside.any(axis=1).sum())))
        del a,outside
    pd.DataFrame(rows).to_csv(saida/'diagnostico_faixas_sinais.csv',index=False,float_format='%.12g')
    pd.DataFrame(brief).to_csv(saida/'diagnostico_resumo.csv',index=False)
    print(pd.DataFrame(brief).to_string(index=False))

if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--hai21',type=Path,required=True);a.add_argument('--hai22',type=Path,required=True)
    a.add_argument('--saida',type=Path,default=Path('resultados'))
    v=a.parse_args();conferir(v.hai21,v.hai22,v.saida)
