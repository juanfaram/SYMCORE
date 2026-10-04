# SYMCORE v0.3 — Evolution Engine

Motor evolutivo online, reproducible y acotado.

**Capacidades:** genoma mutable (learning rate, L2, feature architecture, momentum y clipping), herencia de pesos, población con selección, Page-Hinkley para drift, promoción champion/challenger mediante comparación pareada con intervalo conservador, genealogía completa, checkpoints y artefactos reproducibles en GitHub Actions.

```bash
python main.py
pytest -q
```

El flujo sigue siendo estrictamente causal: predice antes de aprender cada observación. La población máxima y los rangos de mutación están acotados. Ningún candidato puede editar el repositorio ni promocionarse sin la puerta de validación.

## Próximo nivel

v0.4 separará entrenamiento, validación y shadow deployment; añadirá varias fuentes/dominios, ensemble ponderado por rendimiento y un registry persistente de experimentos.
