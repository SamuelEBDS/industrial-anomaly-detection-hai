"""Consolida os CSVs por arquivo e confere as métricas e os episódios salvos."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from comparar_periodos import MODELOS

def consolidar(root):
    ordem=['J1','J2','J3','EXT']
    def ler(stem):
        return pd.concat([pd.read_csv(root/f'{stem}_{j}.csv',dtype={'versao':str}) for j in ordem],ignore_index=True)
    def salvar(df,name):
        df.to_csv(root/name,index=False,float_format='%.12g')
    m=ler('metricas');e=ler('episodios');p=ler('periodos')
    keys=['rodada','versao','arquivo','modelo']
    assert len(m)==40 and not m.duplicated(keys).any() and not e.duplicated(keys+['episodio']).any()
    for _,row in m.iterrows():
        g=e
        for c in keys: g=g.loc[g[c].eq(row[c])]
        assert (g.sinalizados<=g.registros).all()
        assert len(g)==row.episodios_total and g.sinalizados.sum()==row.VP and g.registros.sum()==row.registros_ataque
        assert g.atraso_segundos.notna().sum()==row.episodios_sinalizados
        assert row.FP+row.VN==row.normais and row.VP+row.FN==row.registros_ataque
        assert row.normais+row.registros_ataque==row.registros
        assert np.isclose(row.f1,2*row.VP/(2*row.VP+row.FP+row.FN))
    cm=[]
    for (j,ver,arq),g in m.groupby(keys[:3],sort=False):
        trio=g.set_index('modelo')
        for n in ['VP','FP','episodios_sinalizados']:
            assert trio.at['Ensemble OU',n] >= max(trio.at['Regressão Logística',n],trio.at['IF com PCA',n])
        assert g.episodios_total.nunique()==g.registros.nunique()==1
    for j in ordem:
        conf=json.loads((root/f'protocolo_{j}.json').read_text(encoding='utf-8'))
        for mod in MODELOS:
            cm.append({'rodada':j,'modelo':mod,'sinais':len(conf['sinais']),'componentes_pca':conf['pca_componentes'] if mod in ('IF com PCA','Ensemble OU') else None,'limiar':conf['limiares'].get(mod),'f1_validacao':conf['f1_validacao'][mod]})
    ext=m[(m.rodada=='EXT')&(m.versao=='22.04')]
    ee=e[(e.rodada=='EXT')&(e.versao=='22.04')]
    out=[]
    for md in MODELOS:
        q=ext[ext.modelo==md]
        assert len(q)==4
        tp,fp,fn,tn=[int(q[c].sum()) for c in ('VP','FP','FN','VN')]
        ep=q.episodios_total.sum();total=tp+fp+fn+tn;h=ee[ee.modelo==md].atraso_segundos.dropna()
        assert int(total)==361200 and int(ep)==58 and tp+fn==12030 and tn+fp==349170
        out.append(dict(rodada='EXT',versao='22.04',arquivo='total de 4 arquivos',modelo=md,registros=total,
                        normais=tn+fp,registros_ataque=tp+fn,VP=tp,FP=fp,FN=fn,VN=tn,precisao=tp/(tp+fp) if tp+fp else 0.,
                        recall=tp/(tp+fn),f1=2*tp/(2*tp+fp+fn),taxa_fp=fp/(fp+tn),episodios_total=ep,
                        episodios_sinalizados=len(h),atraso_mediano_s=float(h.median()) if len(h) else None))
    ex=pd.DataFrame(out)
    assert set(m[m.rodada!='EXT'].groupby('rodada').episodios_total.first().to_dict().items())=={('J1',8),('J2',5),('J3',12)}
    salvar(m,'metricas_por_periodo.csv');salvar(e,'episodios_todos.csv');salvar(p,'inventario_dos_periodos.csv')
    salvar(ex,'externo_agregado.csv');salvar(pd.DataFrame(cm),'limiares_e_validacao.csv')
    return m,e,p,ex

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--saida',type=Path,default=Path('resultados'))
    a=ap.parse_args(); m,e,p,ex=consolidar(a.saida)
    print(m.pivot(index='modelo',columns=['rodada','versao','arquivo'],values='f1').round(4))
    print(ex[['modelo','f1','precisao','recall','FP','taxa_fp','episodios_sinalizados','episodios_total']].to_string(index=False))
