# SYMCORE — self-improving experiment v0.1

Primer experimento real: aprendizaje online sobre demanda horaria de bicicletas.

## Qué hace

- Descarga automáticamente el dataset público **UCI Bike Sharing** (17.379 horas).
- Procesa las observaciones estrictamente en orden: **predecir -> medir error -> aprender**.
- Entrena cinco candidatos simultáneos con estructuras/tasas de aprendizaje distintas.
- Cada 500 observaciones compara el MAE de la ventana reciente.
- Si otro candidato es mejor, cambia automáticamente de campeón.
- Guarda la historia de evolución y ranking final en `evolution.json`.

No modifica su propio código todavía. La auto-mejora de v0.1 es selección adaptativa de estructura/configuración sobre datos no vistos.

## Ejecutar

```bash
python3 experiments/self_improving_v01/main.py
```

Solo requiere Python 3.10+ y la librería estándar.

## Fuente de datos

UCI Bike Sharing Dataset, copia pública:
https://github.com/KeithJLZ/UCI-Bike-Sharing-Dataset

Objetivo: predecir `cnt`, el número total de alquileres de la hora.

## Próximo crecimiento

v0.2: generar candidatos nuevos a partir del campeón, detección explícita de drift y promoción con guardas estadísticas.
