#!/usr/bin/env python3
"""Reconstrução independente do ensaio RF 1:20; NÃO é o script histórico.
Executar na raiz de tcc-hai_modular. Não altera arquivos de execuções anteriores.
"""
import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

SEMENTES = (13, 42, 2026)
N = 10671
TAG = 'P1_B3005'
ESPERADO = {
    'original': {'fp': 13379, 'f1': .17811629780182117, 'recall': .6985147756851937},
    'habitual': {'fp': 12790.333333333334, 'f1': .1773773557133125, 'recall': .6692696371152963},
    'baixa': {'fp': 2898, 'f1': .43287294977768814, 'recall': .6438523962639718},
}

def exigir(condicao, mensagem):
    if not condicao:
        raise ValueError(mensagem)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raiz', type=Path, default=Path.cwd())
    p.add_argument('--auditar', action='store_true', help='Confere dados e FP; não treina modelos nem grava resultados.')
    p.add_argument('--habitual', nargs=2, type=float, metavar=('MIN', 'MAX'), help='Faixa explícita do controle; não recuperada do original.')
    args = p.parse_args()
    raiz = args.raiz.resolve()
    sys.path.insert(0, str(raiz / 'src'))
    from hai_tcc.data import ler_particao, hash_arquivo
    from hai_tcc.features import separar_entradas, selecionar_colunas_treino
    from hai_tcc.sampling import selecionar_normais_treino
    from hai_tcc.models import treinar_rf
    from hai_tcc.evaluation import selecionar_limiar
    from hai_tcc.classificacao.avaliacao import preparar_avaliacao, avaliar
    origem = json.loads((raiz/'data/origem.json').read_text())
    pasta = raiz/'data/hai-21.03'
    arquivos = ['test1.csv.gz','test2.csv.gz','test3.csv.gz','test4.csv.gz','test5.csv.gz','train1.csv.gz','train2.csv.gz']
    for nome, info in origem['arquivos'].items():
        exigir(hash_arquivo(pasta/nome)==info['sha256'], f'Hash divergente: {nome}')
    print('Integridade do dataset: OK', flush=True)
    X, y, meta = separar_entradas(ler_particao(pasta, arquivos[:3]))
    X, colunas, constantes = selecionar_colunas_treino(X)
    exigir(len(X)==270003 and int(y.sum())==5613 and len(colunas)==60 and len(constantes)==19, 'Treino diferente do protocolo')
    Xv, yv, mv = separar_entradas(ler_particao(pasta, [arquivos[3]], colunas), colunas)
    Xt, yt, mt = separar_entradas(ler_particao(pasta, [arquivos[4]], colunas), colunas)
    exigir((len(Xv),int(yv.sum()),len(Xt),int(yt.sum()))==(39601,1157,92401,2177), 'Validação/teste divergente')
    base, intervalos = preparar_avaliacao(mt,yt)
    exigir(len(intervalos)==12, 'Número de episódios diferente')
    arquivo_ref = raiz/'runs/baseline_modular/previsoes_e_metricas/teste_previsoes.csv.gz'
    ref = pd.read_csv(arquivo_ref, parse_dates=['time'])
    chaves = ['time','arquivo_origem','indice_origem','attack_real']
    pd.testing.assert_frame_equal(base[chaves].reset_index(drop=True),ref[chaves].reset_index(drop=True),check_dtype=False)
    fp = yt.eq(0) & ref['previsao_rf'].eq(1)
    minimo = float(X.loc[y.eq(0),TAG].min())
    fp_baixa = int((fp & Xt[TAG].lt(minimo)).sum())
    top5 = int(mt.loc[fp,'time'].dt.floor('h').value_counts().head(5).sum())
    print(f'RF ponderado: FP={int(fp.sum())}; cinco horas={top5}; abaixo do mínimo normal ({minimo:.5f} L/h)={fp_baixa}')
    print('Referências do artigo: 10718 FP; 9791 em cinco horas; 8984 abaixo do mínimo.')
    print('Auditoria FP:', 'OK' if (int(fp.sum()),top5,fp_baixa)==(10718,9791,8984) else 'DIVERGÊNCIA (investigar)')
    X1,y1,m1=separar_entradas(ler_particao(pasta,[arquivos[5]],colunas),colunas)
    X2,y2,m2=separar_entradas(ler_particao(pasta,[arquivos[6]],colunas),colunas)
    exigir(y1.eq(0).all() and y2.eq(0).all(), 'train1/train2 não exclusivamente normais')
    elegivel=y1.eq(0) & ~m1.time.isin(meta.time) & m1.time.lt(mt.time.min())
    grupos={'baixa': X1.index[elegivel & X1[TAG].ge(850) & X1[TAG].lt(950)]}
    if args.habitual:
        lo,hi=args.habitual
        exigir(lo<hi and lo>=950, 'Informe faixa habitual válida e sem sobreposição à faixa baixa')
        grupos['habitual']=X1.index[elegivel & X1[TAG].ge(lo) & X1[TAG].lt(hi)]
    for nome, idx in grupos.items():
        print(f'{nome}: {len(idx)} candidatos elegíveis em train1')
    if args.auditar:
        print('Auditoria somente leitura concluída. Nenhum treinamento iniciado.')
        return
    for nome,idx in grupos.items():
        exigir(len(idx)>=N,f'{nome}: menos de {N} candidatos; não altere o critério para forçar os números')
    print('ATENÇÃO: reconstrução; estratégia exata de troca e versão original do ensaio não confirmadas.')
    out=raiz/'runs'/('reconstrucao_vazao_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True,exist_ok=False)
    protocolo={'natureza':'reconstrucao independente, nao script historico','python':platform.python_version(),'sklearn':sklearn.__version__,'numpy':np.__version__,'pandas':pd.__version__,'seeds':SEMENTES,'n_substituicoes':N,'faixa_baixa':[850,950],'faixa_habitual':args.habitual,'regra_faixas':'min inclusivo; max exclusivo','regra_troca':'sorteio uniforme sem reposicao de normais da amostra e de candidatos train1; ordem dos originais preservada e novos exemplos anexados','origem_dados':origem,'codigo_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rf':{'n_estimators':100,'class_weight':None,'random_state':42,'n_jobs':-1},'criterio_limiar':'maximo F1 em test4; desempate maior corte'}
    (out/'protocolo.json').write_text(json.dumps(protocolo,ensure_ascii=False,indent=2))
    linhas=[]
    for s in SEMENTES:
        amostra=selecionar_normais_treino(y,meta,20,s)
        exigir(int(y.loc[amostra].sum())==5613 and len(amostra)==117873,'Amostra RF 1:20 incorreta')
        normais=amostra[y.loc[amostra].eq(0).to_numpy()]
        remover=np.random.default_rng(s).choice(normais.to_numpy(),N,replace=False)
        manter=amostra[~amostra.isin(remover)]
        limiar_original=None
        for nome in ['original',*grupos]:
            if nome=='original':
                xx,yy=X.loc[amostra],y.loc[amostra]
                ident=meta.loc[amostra].copy()
            else:
                novos=np.sort(np.random.default_rng(s+1).choice(grupos[nome].to_numpy(),N,replace=False))
                xx=pd.concat([X.loc[manter],X1.loc[novos]],ignore_index=True)
                yy=pd.concat([y.loc[manter],y1.loc[novos]],ignore_index=True)
                ident=pd.concat([meta.loc[manter],m1.loc[novos]],ignore_index=True)
            exigir(len(xx)==117873 and int(yy.sum())==5613,'Tamanho ou ataques mudaram')
            exigir(not ident.duplicated(['arquivo_origem','indice_origem']).any(),'Identidades duplicadas')
            ident.assign(attack=yy.to_numpy()).to_csv(out/f'amostra_{nome}_{s}.csv.gz',index=False)
            print(f'Treinando {nome}, semente {s}',flush=True)
            modelo,_=treinar_rf(xx,yy,protocolo['rf'])
            classe=list(modelo.classes_).index(1)
            corte=selecionar_limiar(yv,modelo.predict_proba(Xv)[:,classe])
            if nome=='original': limiar_original=corte
            scores=modelo.predict_proba(Xt)[:,classe]
            scores2=modelo.predict_proba(X2)[:,classe]
            for criterio,lim in [('recalibrado_test4',corte),('corte_original',limiar_original)]:
                met,_,eps,pred=avaliar(base,intervalos,scores,lim)
                linhas.append({'configuracao':nome,'semente':s,'criterio':criterio,'fp_train2':int((scores2>=lim).sum()),'taxa_fp_train2':float((scores2>=lim).mean()),**met})
                base.assign(pontuacao=scores,alerta=pred).to_csv(out/f'previsoes_{nome}_{s}_{criterio}.csv.gz',index=False)
                eps.to_csv(out/f'episodios_{nome}_{s}_{criterio}.csv',index=False)
            pd.DataFrame(linhas).to_csv(out/'metricas_por_semente.csv',index=False)
    tabela=pd.DataFrame(linhas)
    medias=tabela[tabela.criterio.eq('recalibrado_test4')].groupby('configuracao').agg(fp=('falsos_positivos','mean'),f1=('f1','mean'),recall=('recall','mean'),taxa_fp_train2=('taxa_fp_train2','mean'))
    medias.to_csv(out/'medias.csv')
    print('\nMédias da RECONSTRUÇÃO, não os resultados históricos:')
    print(medias.to_string())
    print('\nComparação com os registros documentados:')
    for nome,r in medias.iterrows():
        e=ESPERADO[nome]
        ok=(round(r.fp)==round(e['fp']) and round(r.f1,4)==round(e['f1'],4) and round(r.recall*100,2)==round(e['recall']*100,2))
        print(nome, 'COINCIDE nas casas publicadas' if ok else 'DIVERGÊNCIA — não substituir o artigo')
    print('Resultados novos, separados:',out)

if __name__=='__main__':
    main()
