# Plots

Todos los PDF se regeneran desde los CSV de `results/csv/` (comandos desde la raíz del paquete).

| Carpeta | Contenido | Generado por |
|---|---|---|
| `01_convergence/` | `signorini_k*`: ‖σ−σ_h‖, ‖u−u_h‖ (benchmark suave, uniforme). `postprocess_k*`: ‖u−u_h‖, ‖u−ν_h‖, ‖σ−σ_h‖ | `python3 studies/convergence/plot_convergence.py` |
| `02_stenberg/` | `stenberg_k*`: ν_h de (3.2) vs ũ_h de (2.10) (Prop. 3.1). Usa ‖∇_h(u−·)‖: σ_h es el mismo para ambos postprocesos | ídem |
| `03_aposteriori/` | `aposteriori_{smooth,singular}`: efectividad η̃/E y saturación δ, Signorini vs lineal (uniforme) | `python3 studies/aposteriori/plot_aposteriori.py` |
| `04_adaptivity/` | `adaptive_{singular,offset}_*`: adaptativo vs uniforme, mallas, Signorini vs lineal, ‖σ−σ_h‖ | `python3 studies/adaptivity/plot_adaptive.py` |
| `05_references/` | Ejemplos de Führer–Heuer–Stephan (`fhs42`, `fhs43`) y Bartels–Gudi–Kaltenbach (`bgk72`, `bgk731`): errores y η̃, efectividad, mallas adaptativas | `python3 studies/references/plot_references.py` |
| `06_estimators/` | Estimador η (variante B, `contact_estimator`) vs η̃: `<ex>_effectivity` (η/E_s y η̃/E), `<ex>_k*` (errores, η, η̃), `<ex>_drivers` (adaptividad guiada por η vs η̃), `<ex>_components` (peso de cada término en η²), `<ex>_meshes`, `<ex>_estimators` (sin solución exacta). `contact_variants_{singular,offset}`: comparación de los términos de contacto A, B, Bf, Bs, C. `contact_ratio`: término de contacto / error de borde en la zona de contacto. `linear_vs_signorini`: η/E_s con y sin contacto (misma u). `bgk72_oscillation`: η/E_s y osc/E_s con f ∉ P_k | `python3 studies/estimators/plot_estimators.py` |

Convenciones: color = cantidad (azul ‖σ−σ_h‖, naranja ‖u−u_h‖, verde ‖u−ν_h‖, gris η̃);
relleno/línea gruesa = adaptativo o Signorini, hueco/línea fina = uniforme o lineal;
triángulos = orden óptimo.
