import streamlit as st
import pickle
import numpy as np
import pandas as pd

with open('models/modelo_svm.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('models/scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

feature_info = {
    'last_gpa': {
        'description': 'Promedio general acumulado del semestre anterior (sobre 4).',
        'options': {
            1: '<2.00',
            2: '2.00–2.49',
            3: '2.50–2.99',
            4: '3.00–3.49',
            5: '>3.49'
        }
    },
    'mo_edu': {
        'description': 'Nivel educativo de la madre.',
        'options': {
            1: 'Primaria',
            2: 'Secundaria',
            3: 'Preparatoria',
            4: 'Universidad',
            5: 'Maestría',
            6: 'Doctorado'
        }
    },
    'fa_edu': {
        'description': 'Nivel educativo del padre.',
        'options': {
            1: 'Primaria',
            2: 'Secundaria',
            3: 'Preparatoria',
            4: 'Universidad',
            5: 'Maestría',
            6: 'Doctorado'
        }
    },
    'listening': {
        'description': 'Nivel de atención en clase.',
        'options': {
            1: 'Nunca',
            2: 'A veces',
            3: 'Siempre'
        }
    },
    'flip': {
        'description': 'Utilidad de la clase invertida.',
        'options': {
            1: 'Nada útil',
            2: 'Útil',
            3: 'No aplica'
        }
    },
    'scholarship': {
        'description': 'Tipo de beca.',
        'options': {
            1: 'Sin beca',
            2: '25%',
            3: '50%',
            4: '75%',
            5: 'Completa'
        }
    },
    'notes': {
        'description': 'Frecuencia al tomar notas.',
        'options': {
            1: 'Nunca',
            2: 'A veces',
            3: 'Siempre'
        }
    },
    'siblings': {
        'description': 'Número de hermanos/as.',
        'options': {
            1: '1',
            2: '2',
            3: '3',
            4: '4',
            5: '5 o más'
        }
    }
}

st.title("Predicción de desempeño estudiantil")
st.write("Selecciona los valores correspondientes a cada característica para predecir si un estudiante aprobará.")

input_data = []

for feature, info in feature_info.items():
    st.markdown(f"**{feature}**: {info['description']}")
    option_labels = list(info['options'].values())
    option_values = list(info['options'].keys())
    selection = st.selectbox(
        f"Selecciona una opción para {feature}",
        options=option_values,
        format_func=lambda x: f"{info['options'][x]}"
    )
    input_data.append(selection)

if st.button("Predecir"):
    input_array = np.array(input_data).reshape(1, -1)
    input_df = pd.DataFrame(input_array, columns=feature_info.keys())
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.success("El modelo predice que el estudiante **APROBARÁ**.")
    else:
        st.error("El modelo predice que el estudiante **NO aprobará**.")
