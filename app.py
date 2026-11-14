import streamlit as st, numpy as np, json
M = json.load(open('modelo_rna.json'))
def forward(x):
    a = (np.array(x, float) - M['mean']) / np.array(M['std'])
    for l, (W, b) in enumerate(zip(M['W'], M['b'])):
        z = a @ np.array(W) + np.array(b)
        a = np.tanh(z) if l < len(M['W']) - 1 else 1 / (1 + np.exp(-z))
    return float(a[0])
st.set_page_config(page_title='Predictor Diabetes RNA', page_icon='🩺', layout='wide')
st.title('🩺 Predictor de Diabetes — Red Neuronal Artificial'); st.divider()
c1, c2, c3 = st.columns(3)
edad = c1.number_input('Edad', 16, 90, 45); genero = c1.selectbox('Género', ['Masculino (1)', 'Femenino (2)'])
sint = M['features'][2:]; vals = []
for i, s in enumerate(sint): vals.append(int([c1, c2, c3][i % 3].checkbox(s.replace('_', ' '))))
if st.button('🔍 Predecir Riesgo', use_container_width=True):
    prob = forward([edad, 1 if 'Masculino' in genero else 2] + vals)
    (st.error if prob >= 0.5 else st.success)(f"{'⚠️ RIESGO ALTO' if prob>=0.5 else '✅ RIESGO BAJO'} — {prob*100:.1f}%")
    st.progress(prob); st.caption('⚠️ Resultado orientativo. Consulte siempre a un médico.')
