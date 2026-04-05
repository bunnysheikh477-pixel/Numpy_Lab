from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import io
import numpy as np
import sympy as sp
import matplotlib
import matplotlib.pyplot as plt
from scipy import integrate
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, FileResponse
from pydantic import BaseModel


# Use non-interactive backend for plotting
matplotlib.use('Agg')

# Base project directory
BASE_DIR = Path(__file__).resolve().parent

# Symbolic variable
x = sp.symbols('x')

# ------------------------
# Math functions classes
# ------------------------
class Functions:
    @staticmethod
    def parse(expr_str):
        expr = sp.sympify(expr_str)
        func = sp.lambdify(x, expr, modules=['numpy'])
        return expr, func

    @staticmethod
    def symbolic_derivative(expr):
        return sp.diff(expr, x)

    @staticmethod
    def second_derivative(expr):
        return sp.diff(expr, x, 2)

    @staticmethod
    def symbolic_integration(expr):
        return sp.integrate(expr, x)

    @staticmethod
    def numerical_integration(func, a, b):
        result, error = integrate.quad(func, a, b)
        return result, error

    @staticmethod
    def cagr(B_v, E_v, Y):
        if B_v <= 0 or E_v <= 0 or Y <= 0:
            raise ValueError("Values must be positive")
        return (E_v / B_v) ** (1 / Y) - 1

class MathBook:
    def __init__(self):
        self.functions = Functions()
math_book = MathBook()

# ------------------------
# FastAPI app setup
# ------------------------
app = FastAPI(
    title="Calculus and Analytical Geometry API",
    description="API for derivatives, integration, plotting, and more.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # "*" allows all origins, you can set your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=BASE_DIR / "frontend"), name="static")

@app.get("/")
async def root():
    return FileResponse(BASE_DIR / "frontend" / "index.html")

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


# ------------------------
# Request models
# ------------------------
class DerivativeRequest(BaseModel):
    expr: str

class IntegrationRequest(BaseModel):
    expr: str

class DefiniteIntegralRequest(BaseModel):
    expr: str
    a: float
    b: float

class PlotGraphRequest(BaseModel):
    expr: str
    a: float
    b: float

class CAGRRequest(BaseModel):
    BeginningValue: float
    EndingValue: float
    Years: float

class PieChartRequest(BaseModel):
    crops: list[str]
    yields: list[float]

# ------------------------
# Endpoints
# ------------------------
@app.post("/api/derivative")
async def derivative(req: DerivativeRequest):
    try:
        sym_expr, _ = Functions.parse(req.expr)        # Unpack correctly
        first_der = Functions.symbolic_derivative(sym_expr)
        second_der = Functions.second_derivative(sym_expr)

        return {
            "Expression": req.expr,
            "first_derivative": str(first_der),
            "second_derivative": str(second_der),
            "result": f"{first_der} and {second_der}"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/integration")
async def integration(req: IntegrationRequest):
    try:
        expr, _ = Functions.parse(req.expr)
        integral = Functions.symbolic_integration(expr)
        return {
            "expression": req.expr,
            "integral": str(integral),
            "result": str(integral)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))



@app.post("/api/definite_integral")
async def definite_integral(req: DefiniteIntegralRequest):
    try:
        _, func = Functions.parse(req.expr)
        result, error = Functions.numerical_integration(func, req.a, req.b)
        return {
            "expression": req.expr,
            "a": req.a,
            "b": req.b,
            "result": result,
            "error": error
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/plot_graph")
async def plot_graph(req: PlotGraphRequest):
    try:
        expr_sym, func = Functions.parse(req.expr)
        xs = np.linspace(req.a, req.b, 1000)
        ys = func(xs)

        fig, ax = plt.subplots(figsize=(10,6))
        ax.plot(xs, ys, label=str(expr_sym))
        ax.axhline(0, color='black', lw=0.5, ls='--')
        ax.axvline(0, color='black', lw=0.5, ls='--')
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.set_title('Graph of ' + str(expr_sym))
        ax.legend()
        ax.grid(True)

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return Response(content=buf.getvalue(), media_type='image/png')
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/cagr")
async def calculate_cagr(req: CAGRRequest):
    try:
        rate = Functions.cagr(req.BeginningValue, req.EndingValue, req.Years)
        t = np.linspace(0, req.Years, 200)
        profit = req.BeginningValue * (1 + rate) ** t

        fig, ax = plt.subplots(figsize=(10,6))
        ax.plot(t, profit, 'b-', linewidth=2)
        ax.scatter([0, req.Years], [req.BeginningValue, req.EndingValue], color='red', zorder=5)
        ax.set_xlabel("Year")
        ax.set_ylabel("Value")
        ax.set_title(f"Company Growth (CAGR = {rate:.2%})")
        ax.grid(True)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)

        return {
            "cagr": rate,
            "cagr_percent": f"{rate:.2%}"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/pie-chart")
async def pie_chart(req: PieChartRequest):
    try:
        if len(req.crops) != len(req.yields):
            raise ValueError("Crops and yields must have the same length")

        colors = plt.cm.tab20(np.linspace(0, 1, len(req.yields)))

        fig, ax = plt.subplots(figsize=(8,8))
        ax.pie(req.yields, labels=req.crops, autopct='%1.0f%%', colors=colors)
        ax.set_title("Crops and their Yields")

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return Response(content=buf.getvalue(), media_type='image/png')
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/health")
async def health():
    return {"status": "healthy"}





@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)