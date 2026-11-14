import json
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib import colors
from reportlab.lib.units import cm
d=json.load(open('resultados.json')); R,G=d['rna'],d['gbt']
P=lambda v:f"{v*100:.2f} %"
B=ParagraphStyle('b',fontName='Times-Roman',fontSize=12,leading=24,alignment=TA_JUSTIFY,firstLineIndent=1.27*cm)
C=ParagraphStyle('c',parent=B,alignment=TA_CENTER,firstLineIndent=0)
H=ParagraphStyle('h',parent=C,fontName='Times-Bold')
H2=ParagraphStyle('h2',parent=B,fontName='Times-Bold',firstLineIndent=0)
CAP=ParagraphStyle('cap',parent=B,firstLineIndent=0,leading=14,fontSize=12)
S=ParagraphStyle('s',fontName='Times-Roman',fontSize=9,leading=11)
st=[]
st+= [Spacer(1,1.5*cm),Paragraph('<b>Universidad San Ignacio de Loyola</b>',C),Paragraph('Facultad de Ingeniería e Inteligencia Artificial',C),Spacer(1,1.2*cm),
 Paragraph('<b>Redes Neuronales Artificiales para la Predicción Temprana de Diabetes: Comparación entre un Perceptrón Multicapa y Gradient Boosted Trees</b>',C),Spacer(1,1*cm),
 Paragraph('Actividad 09 (Semana 13) — Grupo 7',C),Spacer(1,.8*cm),
 Paragraph('Morales Silva, Omar Jean Piere — 2313215',C),Paragraph('Maquen Caisan, Fabian — 2311685',C),Paragraph('Alcalá Barzola, Matias Alejandro — 2313378',C),Spacer(1,.8*cm),
 Paragraph('Curso: Agentes Inteligentes',C),Paragraph('Docente: Herrera Quispe, José Alfredo',C),Spacer(1,.8*cm),Paragraph('Lima, 13 de noviembre de 2026',C),PageBreak()]
st+= [Paragraph('Introducción',H),
 Paragraph('El presente informe desarrolla la Opción 2 de la práctica de la semana 13. Se entrenó una red neuronal artificial (RNA) de tipo feedforward sobre el conjunto de datos <i>diabetes_data_es_GOLD</i>, compuesto por 520 pacientes y 16 variables predictoras (edad, género y 14 síntomas binarios), y se comparó con un modelo de Gradient Boosted Trees (GBT). La partición fue 80/20 estratificada con semilla 42 (416 registros de entrenamiento y 104 de prueba). La RNA tiene la arquitectura [16]→[32]→[16]→[1], con activación tanh en las capas ocultas y sigmoide en la salida, y se implementó con <i>MLPClassifier</i> de scikit-learn con parada temprana (Pedregosa et al., 2011). El GBT usó 100 árboles, profundidad máxima 5 y tasa de aprendizaje 0.1.',B),
 Paragraph('Correlación y estadísticas',H),
 Paragraph(f"Las variables con mayor correlación con <i>TieneDiabetes</i> son Poliuria (r = {d['corr']['Poliuria']}), Polidipsia (r = {d['corr']['Polidipsia']}), Género (r = {d['corr']['Genero']}), Pérdida repentina de peso (r = {d['corr']['Perdida_repentina_peso']}) y Paresia parcial (r = {d['corr']['Paresia_parcial']}). La correlación positiva de Género indica que la clase positiva es más frecuente en mujeres (codificación 2), y Alopecia (r = -0.268) se asocia en sentido inverso. Poliuria y polidipsia son los síntomas clásicos de la diabetes, lo cual coincide con el conocimiento médico. Variables como Picazón, Cicatrización lenta u Obesidad presentan correlaciones débiles.",B),
 Paragraph('<b>Figura 1</b>',CAP),Paragraph('<i>Correlación de cada variable con TieneDiabetes</i>',CAP),Image('correlacion.png',width=14*cm,height=10.5*cm),
 Paragraph('Métricas e interpretación: RNA frente a GBT',H),
 Paragraph('<b>Tabla 1</b>',CAP),Paragraph('<i>Métricas en el conjunto de prueba (n = 104)</i>',CAP)]
t=Table([['Modelo','Train acc.','Test acc.','F1','AUC','Kappa'],['RNA [32,16]',P(R['train_acc']),P(R['acc']),f"{R['f1']:.4f}",f"{R['auc']:.4f}",f"{R['kappa']:.4f}"],['Gradient Boosted Trees',P(G['train_acc']),P(G['acc']),f"{G['f1']:.4f}",f"{G['auc']:.4f}",f"{G['kappa']:.4f}"]])
t.setStyle(TableStyle([('FONT',(0,0),(-1,-1),'Times-Roman',11),('FONT',(0,0),(-1,0),'Times-Bold',11),('LINEABOVE',(0,0),(-1,0),1,colors.black),('LINEBELOW',(0,0),(-1,0),.5,colors.black),('LINEBELOW',(0,-1),(-1,-1),1,colors.black)]))
cr,cg=d['cm_rna'],d['cm_gbt']
st+=[t,Spacer(1,.4*cm),
 Paragraph(f"El GBT ganó en todas las métricas: {P(G['acc'])} de accuracy frente a {P(R['acc'])} de la RNA, F1 de {G['f1']:.3f} frente a {R['f1']:.3f} y AUC de {G['auc']:.3f} frente a {R['auc']:.3f}. En la matriz de confusión, la RNA cometió {cr[0][1]} falsos positivos y {cr[1][0]} falsos negativos, mientras que el GBT cometió {cg[0][1]} y {cg[1][0]}, respectivamente. El GBT gana porque los datos son tabulares, casi todos binarios y con pocas filas: los árboles separan directamente reglas del tipo «si tiene poliuria y polidipsia», mientras que la RNA necesita más datos para aprender las mismas interacciones. Aun así, la RNA obtiene un AUC de {R['auc']:.3f}, lo que indica que ordena muy bien a los pacientes por riesgo.",B),
 Paragraph('<b>Figura 2</b>',CAP),Paragraph('<i>Matrices de confusión de la RNA y del GBT</i>',CAP),Image('confusion_matrix.png',width=15*cm,height=6.25*cm),
 Paragraph('<b>Figura 3</b>',CAP),Paragraph(f"<i>Curva de aprendizaje de la RNA ({d['epocas']} épocas con parada temprana)</i>",CAP),Image('curva_aprendizaje.png',width=15*cm,height=5*cm),
 Paragraph('Desafíos realizados',H),Paragraph('<b>Tabla 2</b>',CAP),Paragraph('<i>Registro de experimentos (un parámetro a la vez)</i>',CAP)]
rows=[['Experimento','Train acc.','Test acc.','F1','AUC','Observación']]+[[f['Experimento'],f"{f['train_acc']:.4f}",f"{f['test_acc']:.4f}",f"{f['F1']:.4f}",f"{f['AUC']:.4f}",f['Observacion']] for f in d['desafios']]
t2=Table(rows,colWidths=[5.6*cm,1.9*cm,1.9*cm,1.6*cm,1.6*cm,3.4*cm]); t2.setStyle(TableStyle([('FONT',(0,0),(-1,-1),'Times-Roman',9.5),('FONT',(0,0),(-1,0),'Times-Bold',9.5),('LINEABOVE',(0,0),(-1,0),1,colors.black),('LINEBELOW',(0,0),(-1,0),.5,colors.black),('LINEBELOW',(0,-1),(-1,-1),1,colors.black)]))
sr,sg=d['semillas']['RNA'],d['semillas']['GBT']
import statistics as s
st+=[t2,Spacer(1,.4*cm),
 Paragraph("<b>Desafío 1 (más capas ocultas).</b> Pasar a [64,32,16] subió el accuracy de prueba a 96.15 %, pero agregar una cuarta capa ([128,64,32,16]) lo bajó a 94.23 % y redujo el AUC. Con 416 registros de entrenamiento, más capas añaden parámetros que el modelo no puede aprovechar, así que existe un punto a partir del cual «más capas» empieza a perjudicar.",B),
 Paragraph("<b>Desafío 2 (partición extrema).</b> Con 50/50 la RNA solo entrenó con 260 registros y su accuracy cayó a 89.62 %. Con 95/5 el accuracy fue 96.15 % y el AUC 1.0, pero con solo 26 pacientes de prueba cada error mueve el resultado casi 4 puntos, por lo que esa cifra no es confiable. La partición 80/20 o 70/30 equilibra datos de entrenamiento y validez de las métricas.",B),
 Paragraph(f"<b>Desafíos 3 y 4 (GBT).</b> Con 10 árboles el GBT ya llegó a 98.08 % y con 500 no mejoró, aunque tardó {d['tiempos']['500']:.2f} s frente a {d['tiempos']['10']:.2f} s: hay saturación. Con lr = 0.001 el modelo no aprendió en 100 árboles y predijo todo como positivo (61.54 %), mientras que con lr = 1.0 sobreajustó (100 % en entrenamiento) sin ganar en prueba.",B),
 Paragraph(f"<b>Desafíos 5 y 6.</b> Al quitar Poliuria, el accuracy de la RNA bajó de {P(R['acc'])} a 89.42 %, lo que confirma que es el síntoma más predictivo, igual que en el ranking de <i>feature_importances_</i> del GBT (Figura 4). Con cinco semillas, la RNA promedió {s.mean(sr)*100:.2f} % (desviación {s.pstdev(sr)*100:.2f}) y el GBT {s.mean(sg)*100:.2f} % (desviación {s.pstdev(sg)*100:.2f}), así que el GBT es más preciso y más estable.",B),
 Paragraph('<b>Figura 4</b>',CAP),Paragraph('<i>Importancia de variables del GBT (feature_importances_)</i>',CAP),Image('feature_importances.png',width=13*cm,height=9.75*cm),
 Paragraph('Reflexión: ¿cuándo usar RNA y cuándo GBT?',H),
 Paragraph('Para este problema recomendamos el GBT: obtuvo mejor accuracy, F1 y AUC, varía menos entre semillas, entrena en fracciones de segundo y permite explicar qué síntomas pesan más, algo importante en un contexto médico. La RNA sería preferible con muchos más datos, con variables continuas o no estructuradas (imágenes, señales, texto) o cuando se necesite un modelo diferenciable que se integre en un sistema mayor. Una ventaja práctica de la RNA es que su modelo cabe en un archivo pequeño de pesos y se ejecuta en el navegador sin servidor, como muestra la app web publicada.',B),
 Paragraph('Aplicación web',H),
 Paragraph('Los pesos de la RNA se exportaron a <i>PesosRedNeuronal.pmml</i> y <i>modelo_rna.json</i>. Con ellos se construyó <i>index.html</i>, una página en JavaScript puro que aplica la normalización Z-score, tanh en las capas ocultas y sigmoide en la salida, y muestra el riesgo en rojo o verde con una barra de porcentaje. También se incluye <i>app.py</i> para Streamlit. URL pública: https://jeanpiermorales.github.io/s13-diabetes-rna/',B),
 PageBreak(),Paragraph('Referencias',H)]
RS=ParagraphStyle('r',parent=B,firstLineIndent=-1.27*cm,leftIndent=1.27*cm,alignment=0)
for r in ['Islam, M. M. F., Ferdousi, R., Rahman, S., &amp; Bushra, H. Y. (2020). Likelihood prediction of diabetes at early stage using data mining techniques. En <i>Computer Vision and Machine Intelligence in Medical Image Analysis</i> (pp. 113–125). Springer. https://doi.org/10.1007/978-981-13-8798-2_12',
 'Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., &amp; Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. <i>Journal of Machine Learning Research, 12</i>, 2825–2830.',
 'Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. <i>The Annals of Statistics, 29</i>(5), 1189–1232. https://doi.org/10.1214/aos/1013203451']:
    st.append(Paragraph(r,RS))
def pn(c,doc): c.setFont('Times-Roman',12); c.drawRightString(A4[0]-2.54*cm,A4[1]-1.5*cm,str(doc.page))
SimpleDocTemplate('Informe_S13_RNA_Grupo7.pdf',pagesize=A4,leftMargin=2.54*cm,rightMargin=2.54*cm,topMargin=2.54*cm,bottomMargin=2.54*cm).build(st,onFirstPage=pn,onLaterPages=pn)
