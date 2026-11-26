import sqlite3
import joblib
from features import extraer_features_matricula, clasificar_nivel

conn = sqlite3.connect("bd_piloto.db")
modelo = joblib.load("modelo_riesgo.joblib")

filas = conn.execute(
    "SELECT id_matricula, probabilidad, nivel FROM riesgo ORDER BY id_matricula LIMIT 15"
).fetchall()

print(f"{'ID Mat':<8} | {'BD Prob':<10} | {'API Prob':<10} | {'Diff':<10} | {'BD Nivel':<8} | {'API Nivel':<8}")
print("-" * 68)

for id_mat, prob_bd, nivel_bd in filas:
    X = extraer_features_matricula(conn, id_mat)
    prob_api = float(modelo.predict_proba(X)[0][1])
    nivel_api = clasificar_nivel(prob_api)
    diff = abs(prob_bd - prob_api)
    print(f"{id_mat:<8} | {prob_bd:<10.6f} | {prob_api:<10.6f} | {diff:<10.6f} | {nivel_bd:<8} | {nivel_api:<8}")

conn.close()