# Resumen de pruebas — Megabenchmark GeoJAC

**Fecha:** 2026-08-18 (corrida automatizada) · actualizado 2026-09-03
**Rama:** `develop` (commit `d99edad`) · `main` aún no fusionado — 17 commits atrás, merge pendiente
**Script:** `scripts/run_all_tiers.sh` → `scripts/run_megabenchmark.py` (unattended, checkpoint + commit por tier, RAM guard 1.5 GiB, timeout duro 1800s por corrida)

## 1. Alcance

Comparación de 4 algoritmos de construcción de grafos (`Raw OSM`, `OSMnx`, `GeoJAC`, `NeatNet`) sobre 4 ciudades/distritos de escala creciente, 3 repeticiones por combinación (12 corridas por dataset, 48 en total).

| Tier | Dataset | Escala | Corridas | Estado |
|---|---|---|---|---|
| 1 (control) | Barranco, Lima, Perú | pequeña (~2k nodos) | 12/12 | ✅ OK |
| 2 (medium) | Eixample, Barcelona, España | media (~6.6k nodos) | 12/12 | ✅ OK |
| 2 (medium) | Cercado de Lima, Lima, Perú | media (~11.5k nodos) | 12/12 | ✅ OK |
| 3 (large) | Lima Metropolitana, Perú | grande | 0/12 | ❌ **FAILED — no probado** |
| 4 (extreme) | Cuauhtémoc, Ciudad de México | extrema (~243k nodos) | 3/12 | ⚠️ **Parcial — 3 de 4 algoritmos no probados** |

**Total: 39/48 corridas con resultado válido (81%).**

## 2. Resultados por dataset (mediana de 3 corridas)

| Dataset | Algoritmo | Nodos | Aristas | Long. total (km) | Sinuosidad | Reachability % | Path error p95 | Tiempo (s) | RAM (MB) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Barranco | Raw OSM | 2053 | 2537 | 67.34 | 1.002 | 100.0 | 0.00 | 3.86 | 451 |
| Barranco | OSMnx | 424 | 751 | 67.15 | 1.039 | 99.09 | 20.12 | 5.49 | 468 |
| Barranco | GeoJAC | 582 | 1065 | 64.86 | 1.000 | 99.60 | 66.87 | 5.90 | 477 |
| Barranco | NeatNet | 333 | 638 | 60.37 | 1.017 | 97.13 | 677.24 | 9.57 | 546 |
| Eixample | Raw OSM | 6663 | 7522 | 147.21 | 1.000 | 100.0 | 0.00 | 5.87 | 500 |
| Eixample | OSMnx | 804 | 1460 | 147.21 | 1.008 | 100.0 | 22.72 | 19.86 | 586 |
| Eixample | GeoJAC | 1012 | 1912 | 144.96 | 1.000 | 99.48 | 26.26 | 18.92 | 586 |
| Eixample | NeatNet | 583 | 1158 | 137.12 | 1.014 | 98.50 | 1577.76 | 26.47 | 708 |
| Cercado de Lima | Raw OSM | 11550 | 16845 | 480.88 | 1.002 | 100.0 | 0.00 | 7.77 | 576 |
| Cercado de Lima | OSMnx | 2636 | 5479 | 480.03 | 1.031 | 99.94 | 29.49 | 112.60 | 1617 |
| Cercado de Lima | GeoJAC | 5093 | 10192 | 470.31 | 1.000 | 99.87 | 51.06 | 169.78 | 2474 |
| Cercado de Lima | NeatNet | 2011 | 4702 | 458.60 | 1.019 | 94.46 | 1347.95 | 118.06 | 1612 |
| Cuauhtémoc | Raw OSM | 243145 | 441010 | 18493.20 | 1.002 | 100.0 | 0.00 | 108.36 | 3454 |
| Cuauhtémoc | OSMnx | — | — | — | — | — | — | **TIMEOUT** (1800s) | — |
| Cuauhtémoc | GeoJAC | — | — | — | — | — | — | **TIMEOUT** (1800s) | — |
| Cuauhtémoc | NeatNet | — | — | — | — | — | — | **TIMEOUT** (1800s) | — |

**Lectura rápida:** GeoJAC reduce nodos/aristas de forma consistente frente a Raw OSM manteniendo reachability >99% y errores de path bajos, salvo en Cercado de Lima donde el tiempo/RAM sube fuerte (169.8s / 2.47 GB). NeatNet es el más agresivo en reducción de nodos pero paga el costo más alto en error de path (p95 hasta 1578m en Eixample) y en tiempo de ejecución.

## 3. Casos sin probar

### 3.1 Lima Metropolitana, Perú (Tier 3) — bloqueador externo, 0/12
Los 4 algoritmos fallaron en las 3 repeticiones porque **no se pudo descargar un grafo utilizable**, no por un bug del pipeline:

1. `Lima Metropolitana, Peru` → timeout de descarga a los 600s
2. `Lima, Peru` (fallback) → `ConnectionError: Connection refused` (overpass-api.de caído)
3. `Lima Province, Peru` (fallback) → `ConnectionError: Connection refused`
4. `bbox(-77.2,-12.4,-76.7,-11.7)` (fallback) → `ConnectionError: Connection refused`

Un reintento final logueó `FAILED` con `ParseError: no element found: line 1, column 0` (respuesta vacía/malformada de Overpass) — mismo bloqueador externo, síntoma distinto. Se agotaron los 4 candidatos de fallback disponibles. Detalle completo en `outputs/download_errors.log`. **Decisión tomada: no reintentar hasta que la API de Overpass se estabilice.**

### 3.2 Cuauhtémoc, Ciudad de México (Tier 4) — timeout duro, 3/12
El grafo sí se descargó (243,145 nodos / 441,010 aristas). `Raw OSM` corrió sin problema (~108s), pero `OSMnx`, `GeoJAC` y `NeatNet` excedieron el timeout duro de 1800s en las 3 repeticiones cada uno (9 corridas en `TIMEOUT`). A esta escala ninguno de los tres algoritmos de simplificación terminó dentro de la ventana configurada — es el caso pendiente de mayor prioridad para la tesis, ya que es el dataset de escala "extrema" pensado justamente para mostrar comportamiento límite.

## 4. Datos suplementarios (fuera del pipeline automatizado)

Corridas manuales aportadas por un compañero desde otra máquina (`Test.zip`, recibido 2026-09-03), ya organizadas en `outputs/`:

- **`outputs/bucaramanga_colombia/`** — ciudad nueva, sin corrida previa propia. Añade un quinto dataset de referencia.
- **`outputs/barranco_lima_peru_ext/`** — corrida independiente de Barranco, **distinta** de `outputs/barranco_lima_peru/` (la del pipeline propio). Se guardó aparte a propósito: la fila `con_blindaje / NeatNet` difiere de forma no trivial (`path_error_median` 42.62 vs 3.58, `path_error_p95` 677.24 vs 20.56), y la versión propia trae un plot extra (`sinuosity_coords.png`) que la del compañero no tiene. Pendiente decidir si se reconcilian o se documentan como dos corridas independientes.

Estos datos no pasaron por el `run_all_tiers.sh` unificado (RAM guard, timeout duro, resumability) — tratarlos como referencia, no como parte de la serie comparable del benchmark principal.

## 5. Pendientes

- [ ] Merge `develop → main` (17 commits atrás; bloqueado antes por falta de agente SSH, no reintentado desde entonces).
- [ ] Reintentar Lima Metropolitana cuando Overpass esté estable, o definir una fuente de datos alterna (extracto local `.pbf`, por ejemplo).
- [ ] Subir el timeout (o perfilar/optimizar) para que OSMnx/GeoJAC/NeatNet puedan completar sobre Cuauhtémoc.
- [ ] Reconciliar o documentar formalmente las dos corridas de Barranco (propia vs. compañero).
- [ ] Limpiar archivos sueltos detectados en el repo sin relación con el benchmark: `src/geojac/core/semantics.py` (borrado sin commit) y `src/geojac/core/semantic:q.py` (nombre sospechoso de typo de editor).
