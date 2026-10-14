# KARE-E

Asistente de evidencia para médicos de cuidados paliativos oncológicos, en
español. Responde preguntas clínicas con afirmaciones breves, cada una citada
a una página de una guía o norma, verificadas antes de mostrarse. Cuando el
corpus no alcanza, se abstiene y lo dice.

Es el módulo E (evidencia) de KARE, el proyecto de investigación de Carlos
Orrego (UNAL Manizales) y Karen (Oncólogos de Occidente). El módulo R
(borradores de historia clínica desde el dictado) no existe todavía: llegará
solo dentro de la institución, con aprobación ética e institucional.

## Qué es y qué no es

- Es apoyo a la decisión para profesionales: describe lo que dice la
  evidencia, con la página para comprobarlo.
- No es un dispositivo de diagnóstico, no habla con pacientes ni familias y
  no da indicaciones individualizadas para un caso.
- No recibe datos de pacientes. Una guardia bloquea cualquier pregunta que
  parezca traer identificadores antes de que salga del equipo.

## Cómo responde

```
pregunta
  → guardia de identificadores ............ bloquea; no sale nada del equipo
  → plan de búsqueda (es + en)
  → BM25 por idioma + fusión RRF
  → umbral de cobertura ................... sin pasajes pertinentes: se abstiene
  → borrador con citas [P1]..[Pn]
  → citas válidas ......................... descarta lo que no cita un pasaje real
  → cifras presentes en la fuente ......... descarta dosis o números inventados
  → verificador independiente ............. ve cada afirmación con sus pasajes, sin la pregunta
  → estado: respondida | parcial | sin respuesta | bloqueada
```

El modelo de lenguaje transforma texto en objetos validados; las reglas
deciden qué se muestra. Si la verificación falla, no se muestra nada.

## Primer uso

Requiere Python 3.11 o superior y [uv](https://docs.astral.sh/uv/).

```bash
make install              # dependencias
cp .env.example .env      # y poner ANTHROPIC_API_KEY (u Ollama local, ver .env.example)
# copiar los PDF autorizados a corpus/fuentes/ con los nombres del manifiesto
uv run kare ingerir
uv run kare estado
uv run kare preguntar "¿Qué opioide se prefiere en insuficiencia renal grave?"
```

Sin clave, `kare preguntar` funciona en modo solo recuperación: muestra los
pasajes más cercanos y no genera texto. Sirve para revisar el corpus.

## Comandos

| Comando | Qué hace |
|---|---|
| `kare estado` | Modelo configurado, transcripción y fragmentos por fuente |
| `kare ingerir [--solo ID ...]` | Indexa las fuentes autorizadas del manifiesto |
| `kare preguntar "..." [--descartadas] [--json]` | Responde con evidencia citada |
| `kare preguntar --audio nota.m4a` | Transcribe la pregunta dictada y responde |
| `kare escanear "..." \| --archivo F` | Revisa si un texto trae identificadores |
| `kare transcribir nota.m4a` | Solo transcribe |
| `kare evaluar correr --banco B --salida D` | Corre KARE-E y el modelo solo; arma la hoja ciega |
| `kare evaluar puntuar --corrida D --calificaciones H ...` | Calcula métricas e informe |

## Corpus y licencias

`corpus/manifest.yaml` declara cada fuente con su licencia. Solo se indexa lo
marcado `abierta` o `uso_interno`; lo `por_verificar` y lo `restringida` se
listan pero no entran. Si una fuente baja de categoría, la siguiente ingesta
completa la retira del índice. Los documentos no se versionan.

Cada fragmento conserva fuente, página y huella del archivo, así que cada cita
apunta a una página exacta de una versión concreta.

## Datos y privacidad

- **Zona personal** (este repositorio, la nube, Telegram más adelante): nunca
  datos de pacientes. La guardia corre siempre y no se desactiva. Detecta
  documentos, teléfonos, correos, fechas exactas, números de historia,
  direcciones y nombres después de "paciente", "doña", "señor"… No detecta
  todo: la regla sigue siendo preguntar en términos generales.
- **Audio**: sale al proveedor de transcripción antes de que la guardia pueda
  leerlo. Solo se dictan preguntas genéricas. Para ElevenLabs, activar la
  exclusión del uso de datos para entrenamiento en la cuenta.
- **Zona clínica** (historias y audio con pacientes): solo dentro de la
  institución, con modelos locales o proveedores con retención cero y
  contrato. No está construida.

## Evaluación (PallQA-es)

Karen escribe el banco de preguntas (`eval/GUIA_PREGUNTAS.md`, plantilla en
`eval/pallqa_es_template.csv`). La corrida compara KARE-E con el mismo modelo
sin corpus:

- `ciego.csv`: dos respuestas por pregunta en orden aleatorio, con el mismo
  formato y la misma frase de abstención. Se califica exactitud (0 a 2), daño
  potencial (ninguno, menor, mayor), si lo correcto era abstenerse y cuál se
  prefiere.
- `citas.csv`: cada afirmación de KARE-E con sus pasajes; se califica después,
  porque revela el brazo.
- `clave.csv` queda aparte hasta el final.

`kare evaluar puntuar` reporta exactitud, daño, abstención adecuada, precisión
de citas, preferencia y acuerdo entre calificadoras (kappa de Cohen), con los
criterios provisionales de salida: ningún daño mayor, precisión de citas de al
menos 0,80 y abstención adecuada de al menos 0,80. Los umbrales se fijan antes
de la ronda, no después de ver los resultados.

Limitación conocida: el cegamiento es parcial; el estilo de una respuesta
puede delatar su origen.

## Configuración

Todo por variables de entorno o `.env` (ver `.env.example`). Lo principal:

| Variable | Por defecto | Para qué |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` | `anthropic`, `openai_compat` (OpenAI, Groq, Ollama local) o `none` |
| `ANTHROPIC_MODEL` | `claude-opus-5-5` | Generador y, si no se indica otro, verificador |
| `VERIFIER_MODEL` | vacío | Un modelo distinto para verificar |
| `MIN_COVERAGE` | `0.3` | Fracción de términos de la consulta que debe tener un pasaje |
| `KEEP_PARTIAL_CLAIMS` | `false` | Mostrar afirmaciones parcialmente sustentadas, marcadas |
| `ELEVENLABS_STT_MODEL` | `scribe_v2` | Transcripción de preguntas dictadas |

## Estado

v0.1 (octubre de 2026): pipeline completo, guardia, ingesta con licencias,
transcripción y arnés de evaluación, con 89 pruebas sobre un modelo simulado.
Falta lo que no se puede simular: la primera corrida con un modelo real, el
corpus real y las preguntas de Karen.

Plan de cuatro semanas:

1. Corpus con licencias verificadas, 30 a 50 preguntas de Karen y búsqueda
   sistemática de trabajos parecidos.
2. Primera corrida real; ajuste de umbrales; interfaz por Telegram.
3. Factualidad conforme (P4) sobre `support_score`, KARE-E frente al modelo
   solo y taxonomía de errores.
4. Ronda ciega con Karen y una o dos paliativistas; decisión de seguir o no.

## Desarrollo

```bash
make check     # ruff + mypy estricto + pytest
```

Patrón heredado de `insight_engine` (The Velveteen Project): núcleo
determinista, el LLM solo transforma entradas estructuradas en salidas
validadas con Pydantic, SQLite con SQL explícito, httpx sin SDK. Ver
`AGENTS.md`.
