# KARE

## Qué es
Asistente para una médica de cuidados paliativos oncológicos en Colombia,
dentro de un proyecto de investigación (Carlos Orrego, UNAL Manizales, y
Karen, Oncólogos de Occidente). Dos módulos:

- **E (evidencia)**: responde preguntas clínicas con afirmaciones citadas a
  guías y normas, verificadas, y se abstiene cuando el corpus no alcanza.
  Es lo que existe en v0.
- **R (registros)**: borradores de historia clínica paliativa desde el
  dictado. No existe; solo llegará dentro de la institución y con aprobación.

## Filosofía
- Nada de humo: cada afirmación lleva cita o no sale.
- El LLM transforma texto en objetos validados; las reglas deciden qué se
  muestra. El LLM nunca despacha acciones.
- Ante la duda, abstenerse. Un "no sé" correcto vale más que una respuesta
  bonita.
- Preferir reglas deterministas cuando basten: guardia, cifras, umbrales.
- Nada de sobreingeniería: sin LangChain, sin base vectorial mientras BM25
  alcance, sin SDK donde httpx basta.
- Código tipado (mypy estricto), probado y legible.

## Stack
Python 3.11+, Pydantic, pydantic-settings, httpx, SQLite, pypdf,
snowballstemmer, pytest, ruff, mypy.

## Zonas de datos (no negociable)
- **Zona personal** (este repo, la nube, Telegram): nunca datos de pacientes.
  La guardia de identificadores corre antes de cualquier llamada externa y no
  se puede desactivar. Las preguntas bloqueadas se guardan redactadas; el
  dato detectado no se repite en ningún mensaje.
- El audio sale al proveedor de transcripción antes de que la guardia lo
  lea: solo se dictan preguntas genéricas.
- **Zona clínica** (historias, audio con pacientes): solo dentro de la
  institución, con modelos locales o proveedores con retención cero y
  contrato. No está construida.

## Orden del pipeline (`app/answer/pipeline.py`)
guardia → plan de búsqueda → BM25 es/en + RRF → abstención por cobertura →
borrador con citas → citas válidas → cifras presentes en la fuente →
verificador independiente → estado final.

## Corpus
- `corpus/manifest.yaml` declara cada fuente con licencia; solo se indexa lo
  `abierta` o `uso_interno`.
- Los documentos no se versionan. El banco de preguntas real tampoco, hasta
  publicar (contaminación de modelos futuros).

## Qué evitar
- No debilitar el chequeo de cifras (`app/answer/numbers.py`): es la defensa
  contra la dosis inventada.
- No dirigir respuestas a pacientes o familias.
- No bajar umbrales para "arreglar" abstenciones sin medir en el banco.
- No mostrar afirmaciones si la verificación falla.
- No escribir datos de pacientes en pruebas, ejemplos ni registros: los
  textos de prueba son sintéticos y lo dicen.
