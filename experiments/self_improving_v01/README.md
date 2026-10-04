# SYMCORE — evolutionary online learner v0.2

Experimento real de aprendizaje continuo y evolución automática sobre demanda horaria de bicicletas.

## Ciclo

1. Descarga UCI Bike Sharing (17.379 observaciones horarias).
2. Para cada hora: predice antes de aprender.
3. Mide el error y actualiza los modelos online.
4. El campeón genera tres mutaciones periódicamente o cuando se detecta drift.
5. Los hijos heredan pesos del campeón y mutan learning rate, regularización L2 y estructura de features.
6. La población queda acotada: campeón + mejores rivales + descendientes.
7. Un rival solo reemplaza al campeón si mejora al menos un 2% su MAE reciente.
8. Toda mutación/promoción queda registrada en `evolution.json`.

La v0.2 ya **genera configuraciones que no estaban enumeradas de antemano**. Sigue manteniendo límites explícitos: no reescribe código arbitrario.

## Ejecutar

```bash
cd experiments/self_improving_v01
python3 main.py
```

Opcional:

```bash
python3 main.py --interval 250 --seed 42
```

Solo requiere Python 3.10+ y la librería estándar.

## Fuente

UCI Bike Sharing Dataset (copia pública):
https://github.com/KeithJLZ/UCI-Bike-Sharing-Dataset

Objetivo: predecir `cnt` (alquileres totales por hora) sin mirar datos futuros.

## Próximo salto

v0.3: validación estadística challenger/champion, mutación de conjuntos de features más granular, checkpoints persistentes y evaluación walk-forward reproducible contra baselines.
