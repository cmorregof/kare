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
  → BM25 por idioma + fusión RRF .......... el mismo peso para cada idioma
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
| `kare descargar [--solo ID ...]` | Baja de su enlace oficial las fuentes autorizadas, si coinciden con su huella |
| `kare ingerir [--solo ID ...]` | Indexa las fuentes autorizadas del manifiesto |
| `kare preguntar "..." [--descartadas] [--json]` | Responde con evidencia citada |
| `kare preguntar --audio nota.m4a` | Transcribe la pregunta dictada y responde |
| `kare escanear "..." \| --archivo F` | Revisa si un texto trae identificadores |
| `kare transcribir nota.m4a` | Solo transcribe |
| `kare bot` | Bot privado de Telegram (ver abajo) |
| `kare evaluar correr --banco B --salida D` | Corre KARE-E y el modelo solo; arma la hoja ciega |
| `kare evaluar puntuar --corrida D --calificaciones H ...` | Calcula métricas e informe |

## Bot de Telegram

`kare bot` atiende por Telegram con el mismo pipeline. Es zona personal:
Telegram guarda los mensajes en sus servidores antes de que el bot los lea.
Por eso:

- Solo responde a los IDs de Telegram autorizados. A cualquier otro le dice
  su ID y nada más.
- La guardia corre antes del modelo, sobre todo texto: preguntas, comandos y
  pies de foto. Si encuentra identificadores, el bot borra el mensaje del chat,
  no lo procesa y no repite el dato. Si Telegram no le deja borrarlo, lo dice
  para que lo borres tú.
- Borra sin procesar los textos de más de 600 caracteres, porque una pregunta
  así de larga suele ser una nota clínica pegada. También borra los pedazos en
  que Telegram parte un texto muy largo, que llegan como mensajes aparte.
- No procesa imágenes, documentos ni otros archivos: los borra.
- Una nota de voz llega al servicio de transcripción antes de que la guardia
  lea el texto. Por eso solo se dictan preguntas generales.
- Los registros del servidor no llevan el texto de preguntas, respuestas ni
  transcripciones: solo estados, conteos y tiempos.
- `/start` explica las reglas, `/fuentes` lista los documentos indexados e
  `/id` le dice a cada quien su ID.

### Despliegue en Railway

El repositorio trae un `Dockerfile`; el bot usa *long polling*, así que no
necesita dominio público.

1. En Telegram, `/newbot` con @BotFather. Guarda el token: va solo en Railway.
   De paso, `/setjoingroups` → *Disable*, para que nadie lo meta en un grupo.
2. En Railway: *New Project → Deploy from GitHub repo* → este repositorio.
3. Al servicio, agrégale un volumen con *mount path* `/data`. Ahí quedan los
   PDF y el índice (`CORPUS_DIR=/data/fuentes`, `DB_PATH=/data/kare.db` ya
   vienen en la imagen).
4. Variables del servicio: `ANTHROPIC_API_KEY`, `TELEGRAM_BOT_TOKEN` y
   `TELEGRAM_ALLOWED_USERS` (vacía al principio). Opcional:
   `ELEVENLABS_API_KEY` para notas de voz.
5. Despliega. Al arrancar baja las fuentes con enlace (`kare descargar`),
   las indexa (unos 20 segundos) y queda escuchando. Los registros dicen
   qué se bajó y cuántos fragmentos quedaron.
6. Escríbele `/id` al bot desde cada teléfono autorizado, pon los IDs en
   `TELEGRAM_ALLOWED_USERS` separados por comas y deja que redespliegue.

Si una fuente no baja (el sitio rechaza al servidor o cambió el archivo), el
bot arranca igual con las demás y el registro dice cuál faltó.

## Corpus y licencias

`corpus/manifest.yaml` declara cada fuente con su licencia. Solo se indexa lo
marcado `abierta` o `uso_interno`; lo `por_verificar` y lo `restringida` se
listan pero no entran. Si una fuente baja de categoría, la siguiente ingesta
completa la retira del índice. Los documentos no se versionan.

Una fuente con `download_url` y `sha256` se puede bajar con `kare descargar`,
y solo se acepta si su huella coincide con la de la versión revisada; la
ingesta tampoco indexa un archivo con otra huella. Si la fuente publica otra
versión, alguien la revisa y actualiza `sha256` en el manifiesto.

Cada fragmento conserva fuente, página y huella del archivo, así que cada cita
apunta a una página exacta de una versión concreta.

## Datos y privacidad

- **Zona personal** (este repositorio, la nube, el bot de Telegram): nunca
  datos de pacientes. La guardia corre siempre y no se desactiva. Detecta
  documentos (con puntos, comas o letras), celulares y fijos, correos, fechas
  exactas (numéricas, abreviadas o en palabras), números de historia,
  direcciones (incluidas manzana y casa) y nombres después de "paciente",
  "doña", "señor", "Nombre:"… No detecta un nombre escrito en minúsculas ni uno
  sin palabra que lo anuncie: la regla sigue siendo preguntar en términos
  generales.
- **Audio**: sale al proveedor de transcripción antes de que la guardia pueda
  leerlo. Solo se dictan preguntas genéricas; si la transcripción trae
  identificadores, se muestra redactada y no va al modelo. Para ElevenLabs,
  activar la exclusión del uso de datos para entrenamiento en la cuenta;
  `ELEVENLABS_ZERO_RETENTION=true` pide retención cero (solo cuentas
  Enterprise).
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
| `ELEVENLABS_ZERO_RETENTION` | `false` | Retención cero en ElevenLabs (Enterprise) |
| `TELEGRAM_BOT_TOKEN` | vacío | Token del bot (de @BotFather) |
| `TELEGRAM_ALLOWED_USERS` | vacío | IDs de Telegram autorizados, separados por comas |
| `TELEGRAM_MAX_QUESTION_CHARS` | `600` | Largo máximo de una pregunta en el bot |

## Estado

v0.2.1: una revisión independiente del bot encontró cinco fallas, ninguna
crítica, y están corregidas. El bot decía que había borrado una nota de voz
aunque el borrado fallara. Un texto largo que Telegram partía en dos dejaba
pasar la cola. Los errores de algunos proveedores podían copiar parte de la
pregunta al registro. El respaldo de Whisper podía mandar la clave de otro
proveedor a OpenAI. Una descarga lenta podía colgar el arranque.

v0.2.0 (14 de octubre de 2026): primera corrida con un modelo real y el corpus
real (OMS 2018 y GPC 58). Corrigió tres cosas: encabezados y números de página
dentro de los fragmentos, una fusión de rankings que dejaba por fuera la guía
en inglés, y respuestas parciales presentadas como completas. Suma el bot de
Telegram y la descarga de fuentes con huella.

v0.1.1: pipeline completo, guardia, ingesta con
licencias, transcripción y arnés de evaluación, con 181 pruebas sobre un
modelo simulado. Una revisión independiente de v0.1 encontró fugas en la
guardia (una coma después de la cédula bastaba), un camino por el que texto
libre del modelo llegaba sin verificar a la pantalla y errores de mil veces en
el chequeo de cifras; están corregidos y cada caso quedó como prueba. Falta lo
que no se puede simular: la primera corrida con un modelo real, el corpus real
y las preguntas de Karen.

Plan de cuatro semanas:

1. Corpus con licencias verificadas, 30 a 50 preguntas de Karen y búsqueda
   sistemática de trabajos parecidos.
2. Primera corrida real (hecha); ajuste de umbrales; interfaz por Telegram
   (hecha).
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
