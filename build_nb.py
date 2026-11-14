import nbformat as nbf
C=[]
md=lambda s:C.append(nbf.v4.new_markdown_cell(s)); code=lambda s:C.append(nbf.v4.new_code_cell(s))
md("# Semana 13 — Redes Neuronales Artificiales: RNA vs. Gradient Boosted Trees\n**Grupo 7 — Agentes Inteligentes (USIL)**\n- Morales Silva, Omar Jean Piere — 2313215\n- Maquen Caisan, Fabian — 2311685\n- Alcalá Barzola, Matias Alejandro — 2313378\n\nDataset: diabetes_data_es_GOLD.csv (520 pacientes, 17 columnas). La RNA se implementa con `MLPClassifier` de scikit-learn (arquitectura [16]→[32]→[16]→[1], tanh en ocultas, sigmoide en salida).")
code("""import pandas as pd, numpy as np, json, time
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report, cohen_kappa_score
import warnings; warnings.filterwarnings('ignore')""")
md("## 1. Construcción del dataset GOLD (columnas binarias 0/1, Genero 1/2)")
code("""raw = pd.read_csv('diabetes_data_upload.csv')
cols = ['Edad','Genero','Poliuria','Polidipsia','Perdida_repentina_peso','Debilidad','Polifagia','Candidiasis_genital','Vision_borrosa','Picazon','Irritabilidad','Cicatrizacion_lenta','Paresia_parcial','Rigidez_muscular','Alopecia','Obesidad','TieneDiabetes']
raw.columns = cols
df = raw.copy()
df['Genero'] = df['Genero'].map({'Male':1,'Female':2})
for c in cols[2:16]: df[c] = df[c].map({'Yes':1,'No':0})
df['TieneDiabetes'] = df['TieneDiabetes'].map({'Positive':'Positivo','Negative':'Negativo'})
df.to_csv('diabetes_data_es_GOLD.csv', index=False)
print(df.shape); print(df['TieneDiabetes'].value_counts())
df.head()""")
md("## 2. Estadísticas descriptivas")
code("""est = df.describe().T[['mean','std','min','max']].round(3)
est""")
md("## 3. Carga y preprocesamiento (80/20 estratificado, semilla 42)")
code("""le = LabelEncoder(); df['target'] = le.fit_transform(df['TieneDiabetes'])
print(le.classes_)
FEATS = cols[:16]
X = df[FEATS].values.astype(float); y = df['target'].values
def split(test_size=0.2, seed=42):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    sc = StandardScaler().fit(Xtr)
    return sc.transform(Xtr), sc.transform(Xte), ytr, yte, sc
X_train, X_test, y_train, y_test, scaler = split()
print('Train', X_train.shape, '| Test', X_test.shape)""")
md("## 4. Correlación con el objetivo")
code("""corr = df[FEATS+['target']].corr()['target'].drop('target').sort_values()
print(corr.sort_values(ascending=False).round(3))
plt.figure(figsize=(8,6)); corr.plot(kind='barh', color='steelblue')
plt.title('Correlación de cada feature con TieneDiabetes'); plt.tight_layout(); plt.savefig('correlacion.png', dpi=150); plt.show()
plt.figure(figsize=(11,9)); sns.heatmap(df[FEATS+['target']].corr(), cmap='RdBu_r', center=0, annot=True, fmt='.1f', annot_kws={'size':6})
plt.title('Matriz de correlación de Pearson'); plt.tight_layout(); plt.savefig('matriz_correlacion.png', dpi=150); plt.show()""")
md("## 5. Arquitectura y entrenamiento de la RNA ([16]→[32]→[16]→[1])")
code("""def rna(layers=(32,16), seed=42):
    return MLPClassifier(hidden_layer_sizes=layers, activation='tanh', solver='adam', max_iter=300,
                         early_stopping=True, validation_fraction=0.15, n_iter_no_change=15, random_state=seed)
model = rna(); model.fit(X_train, y_train)
print('Épocas:', model.n_iter_, '| capas:', [w.shape for w in model.coefs_])
fig,(a1,a2)=plt.subplots(1,2,figsize=(12,4))
a1.plot(model.loss_curve_, label='Train'); a1.set_title('Pérdida (log-loss)'); a1.set_xlabel('Época'); a1.legend()
a2.plot(model.validation_scores_, label='Val', color='orange'); a2.set_title('Accuracy de validación'); a2.set_xlabel('Época'); a2.legend()
plt.tight_layout(); plt.savefig('curva_aprendizaje.png', dpi=150); plt.show()""")
md("## 6. Comparación RNA vs. Gradient Boosted Trees")
code("""def evaluar(m, Xtr, Xte, ytr, yte):
    p = m.predict_proba(Xte)[:,1]; yp = (p>=0.5).astype(int)
    return dict(train_acc=accuracy_score(ytr, m.predict(Xtr)), acc=accuracy_score(yte, yp), f1=f1_score(yte, yp), auc=roc_auc_score(yte, p), kappa=cohen_kappa_score(yte, yp)), yp
r_rna, yp_rna = evaluar(model, X_train, X_test, y_train, y_test)
gbt = GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42).fit(X_train, y_train)
r_gbt, yp_gbt = evaluar(gbt, X_train, X_test, y_train, y_test)
tabla = pd.DataFrame([r_rna, r_gbt], index=['RNA [32,16]','Gradient Boosted Trees']).round(4)
tabla""")
md("## 7. Matrices de confusión")
code("""fig,(a1,a2)=plt.subplots(1,2,figsize=(12,5))
for ax, yp, t in [(a1,yp_rna,'RNA'),(a2,yp_gbt,'GBT')]:
    sns.heatmap(confusion_matrix(y_test, yp), annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=['Neg','Pos'], yticklabels=['Neg','Pos'])
    ax.set_title(f'Confusión — {t}'); ax.set_xlabel('Predicción'); ax.set_ylabel('Real')
plt.tight_layout(); plt.savefig('confusion_matrix.png', dpi=150); plt.show()
print(classification_report(y_test, yp_rna, target_names=le.classes_))""")
md("## 8. Desafíos (experimentos, un parámetro a la vez)")
code("""filas=[]
def reg(nombre, r, obs=''): filas.append(dict(Experimento=nombre, train_acc=round(r['train_acc'],4), test_acc=round(r['acc'],4), F1=round(r['f1'],4), AUC=round(r['auc'],4), Observacion=obs))
reg('Base RNA: [32,16], 80/20', r_rna, 'Referencia')
for L,n in [((64,32,16),'Desafío 1a: [64,32,16]'),((128,64,32,16),'Desafío 1b: [128,64,32,16]')]:
    m=rna(L).fit(X_train,y_train); reg(n, evaluar(m,X_train,X_test,y_train,y_test)[0], f'{m.n_iter_} épocas')
for ts,n in [(0.5,'Desafío 2a: Partición 50/50'),(0.3,'Desafío 2-: Partición 70/30'),(0.1,'Desafío 2-: Partición 90/10'),(0.05,'Desafío 2b: Partición 95/5')]:
    a,b,c,d,_=split(ts); reg(n, evaluar(rna().fit(a,c),a,b,c,d)[0], f'{len(d)} muestras de test')
tiempos={}
for nt in [10,50,100,200,500]:
    t0=time.time(); g=GradientBoostingClassifier(n_estimators=nt,max_depth=5,learning_rate=0.1,random_state=42).fit(X_train,y_train); tiempos[nt]=time.time()-t0
    if nt in (10,500): reg(f'Desafío 3{"a" if nt==10 else "b"}: GBT {nt} árboles', evaluar(g,X_train,X_test,y_train,y_test)[0], f'{tiempos[nt]:.2f} s')
lrs={}
for lr in [0.001,0.01,0.1,0.3,1.0]:
    g=GradientBoostingClassifier(n_estimators=100,max_depth=5,learning_rate=lr,random_state=42).fit(X_train,y_train); r=evaluar(g,X_train,X_test,y_train,y_test)[0]; lrs[lr]=r['acc']
    if lr in (0.001,1.0): reg(f'Desafío 4{"a" if lr==0.001 else "b"}: GBT lr={lr}', r)
i=FEATS.index('Poliuria'); Xs=np.delete(X_train,i,1); Xts=np.delete(X_test,i,1)
reg('Desafío 5: Sin Poliuria', evaluar(rna().fit(Xs,y_train),Xs,Xts,y_train,y_test)[0])
semillas={'RNA':[], 'GBT':[]}
for s in [0,7,21,42,99]:
    a,b,c,d,_=split(0.2,s)
    rr=evaluar(rna(seed=s).fit(a,c),a,b,c,d)[0]; rg=evaluar(GradientBoostingClassifier(n_estimators=100,max_depth=5,learning_rate=0.1,random_state=s).fit(a,c),a,b,c,d)[0]
    semillas['RNA'].append(rr['acc']); semillas['GBT'].append(rg['acc'])
    if s==0: reg('Desafío 6: RNA semilla 0', rr); reg('Desafío 6: GBT semilla 0', rg)
desafios=pd.DataFrame(filas); print(desafios.to_string())
print('Accuracy vs n árboles (tiempo s):', {k:round(v,3) for k,v in tiempos.items()})
print('Accuracy vs lr:', lrs)
print('Semillas  RNA media %.4f std %.4f | GBT media %.4f std %.4f' % (np.mean(semillas['RNA']),np.std(semillas['RNA']),np.mean(semillas['GBT']),np.std(semillas['GBT'])))
plt.figure(figsize=(6,4)); plt.semilogx(list(lrs),list(lrs.values()),'o-'); plt.title('Accuracy vs learning rate (GBT)'); plt.xlabel('learning_rate'); plt.ylabel('Accuracy test'); plt.tight_layout(); plt.savefig('acc_vs_lr.png',dpi=150); plt.show()""")
md("## 9. Importancia de variables (bonus)")
code("""imp = pd.Series(gbt.feature_importances_, index=FEATS).sort_values()
imp.plot(kind='barh', figsize=(8,6), color='seagreen'); plt.title('feature_importances_ — GBT'); plt.tight_layout(); plt.savefig('feature_importances.png',dpi=150); plt.show()
print(imp.sort_values(ascending=False).round(4))""")
md("## 10. Exportar el modelo (pesos JSON para la app web y PMML)")
code("""pesos = dict(features=FEATS, mean=scaler.mean_.tolist(), std=scaler.scale_.tolist(), W=[w.tolist() for w in model.coefs_], b=[b.tolist() for b in model.intercepts_], hidden='tanh', output='sigmoid')
json.dump(pesos, open('modelo_rna.json','w'))
# PMML NeuralNetwork (PMML 4.4) escrito a mano desde los pesos
L=[]; ids={}
L.append('<?xml version="1.0" encoding="UTF-8"?>\\n<PMML version="4.4" xmlns="http://www.dmg.org/PMML-4_4">\\n<Header description="RNA Diabetes - Grupo 7"/>\\n<DataDictionary>')
for f in FEATS: L.append(f'<DataField name="{f}" optype="continuous" dataType="double"/>')
L.append('<DataField name="TieneDiabetes" optype="categorical" dataType="string"><Value value="Negativo"/><Value value="Positivo"/></DataField></DataDictionary>')
L.append('<NeuralNetwork functionName="classification" activationFunction="tanh"><MiningSchema>'+''.join(f'<MiningField name="{f}"/>' for f in FEATS)+'<MiningField name="TieneDiabetes" usageType="target"/></MiningSchema><NeuralInputs>')
for k,f in enumerate(FEATS):
    ids[(0,k)]=f'0,{k}'; L.append(f'<NeuralInput id="0,{k}"><DerivedField optype="continuous" dataType="double"><NormContinuous field="{f}"><LinearNorm orig="0" norm="{-scaler.mean_[k]/scaler.scale_[k]}"/><LinearNorm orig="1" norm="{(1-scaler.mean_[k])/scaler.scale_[k]}"/></NormContinuous></DerivedField></NeuralInput>')
L.append('</NeuralInputs>')
for li,(W,B) in enumerate(zip(model.coefs_, model.intercepts_), start=1):
    act = ' activationFunction="logistic"' if li==len(model.coefs_) else ''
    L.append(f'<NeuralLayer{act}>')
    for j in range(W.shape[1]):
        L.append(f'<Neuron id="{li},{j}" bias="{B[j]}">'+''.join(f'<Con from="{li-1},{i}" weight="{W[i,j]}"/>' for i in range(W.shape[0]))+'</Neuron>')
    L.append('</NeuralLayer>')
L.append(f'<NeuralOutputs><NeuralOutput outputNeuron="{len(model.coefs_)},0"><DerivedField optype="categorical" dataType="string"><NormDiscrete field="TieneDiabetes" value="Positivo"/></DerivedField></NeuralOutput></NeuralOutputs></NeuralNetwork></PMML>')
open('PesosRedNeuronal.pmml','w').write('\\n'.join(L))
json.dump(dict(rna=r_rna, gbt=r_gbt, desafios=filas, tiempos=tiempos, lrs={str(k):v for k,v in lrs.items()}, semillas=semillas, corr=corr.sort_values(ascending=False).round(3).to_dict(), imp=imp.sort_values(ascending=False).round(4).to_dict(), epocas=int(model.n_iter_), cm_rna=confusion_matrix(y_test,yp_rna).tolist(), cm_gbt=confusion_matrix(y_test,yp_gbt).tolist()), open('resultados.json','w'), indent=1)
print('Exportado: modelo_rna.json, PesosRedNeuronal.pmml')""")
nb=nbf.v4.new_notebook(); nb.cells=C
nb.metadata['kernelspec']={'name':'python3','display_name':'Python 3','language':'python'}
nbf.write(nb,'01_entrenamiento.ipynb')
