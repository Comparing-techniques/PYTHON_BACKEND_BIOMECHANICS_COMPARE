# PYTHON_BACKEND_BIOMECHANICS_COMPARE

This monolith focuses on the analysis and comparison of biomechanical data, such as joint movements, velocity and other relevant parameters.

## Requirements

- Python 3.8 or superior
- Git
- pip (gestor de paquetes de Python)

## Configuration

1. Clone the repository:

```bash
   https://github.com/Comparing-techniques/PYTHON_BACKEND_BIOMECHANICS_COMPARE.git
   cd PYTHON_BACKEND_BIOMECHANICS_COMPARE
```

2. Create and Activate a Virtual Environment

In macOS or Linux:
```bash
- python3 -m venv .pythonapi
- source .pythonapi/bin/activate
```

In Windows (CMD):
```bash
- python -m venv .pythonapi
- .pythonapi\Scripts\activate
```

3. Install the Dependencies

With the virtual environment enabled, install the necessary dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

Once the dependencies are installed, start the development server with:
```bash
- uvicorn app.main:app_principal  --reload 
```