# Backend ML - Instalación rápida

1. **Entrar y crear entorno virtual:**
   ```powershell
   cd backend_ml
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. **Instalar dependencias:**
   ```powershell
   pip install fastapi uvicorn scikit-learn==1.6.1 pandas numpy joblib pydantic
   ```
   *(O con el archivo: `pip install -r requirements.txt`)*

3. **Ejecutar la API:**
   ```powershell
   uvicorn main:app --reload --port 8000
   ```

* **URL base:** `http://127.0.0.1:8000`
* **Docs / pruebas:** `http://127.0.0.1:8000/docs`