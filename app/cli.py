"""Línea de comandos de KARE-E.

kare estado                        configuración y corpus
kare ingerir [--solo ID ...]       indexa las fuentes autorizadas
kare preguntar "..."               responde con evidencia citada
kare preguntar --audio nota.m4a    transcribe la pregunta y responde
kare escanear "..."                revisa si un texto trae identificadores
kare transcribir nota.m4a          solo transcribe
kare evaluar correr --banco eval/pallqa_es.csv --salida runs/r1
kare evaluar puntuar --corrida runs/r1 --calificaciones runs/r1/ciego_karen.csv
"""

import argparse
import asyncio
import logging
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from app import __version__
from app.answer.baseline import BaselineAnswerer
from app.answer.pipeline import EvidencePipeline, PipelineConfig
from app.answer.render import render_text
from app.core.config import Settings, get_settings
from app.corpus.chunking import ChunkingConfig
from app.corpus.ingest import ingest_manifest
from app.corpus.manifest import load_manifest
from app.corpus.store import CorpusStore
from app.eval.bank import BankError, load_bank
from app.eval.run import run_evaluation
from app.eval.score import render_report, score_run
from app.guard.identifiers import describe, redact, scan
from app.llm.client import build_llm
from app.retrieval.retriever import Retriever
from app.transcription import get_transcriber, transcribe_safely

Handler = Callable[[argparse.Namespace, Settings], int]

NO_TRANSCRIBER = (
    "No hay transcripción configurada: falta ELEVENLABS_API_KEY u OPENAI_API_KEY."
)


def _yes(value: bool) -> str:
    return "sí" if value else "no"


def _model_label(settings: Settings) -> str:
    if settings.llm_provider == "anthropic":
        key = bool(settings.anthropic_api_key)
        return f"anthropic · {settings.anthropic_model} · clave: {_yes(key)}"
    if settings.llm_provider == "openai_compat":
        model = settings.openai_model or "(falta OPENAI_MODEL)"
        return f"openai_compat · {model} · {settings.openai_base_url}"
    return "ninguno (solo recuperación)"


AUDIO_ALREADY_SENT = (
    "Aviso: la transcripción trae datos que identifican a una persona. El audio ya "
    "salió al servicio de transcripción; el texto no se enviará al modelo. Dicta "
    "solo preguntas generales."
)


def _shown_transcript(text: str) -> str:
    """La transcripción tal cual, o redactada (con aviso) si trae identificadores."""
    findings = scan(text)
    if not findings:
        return text
    print(AUDIO_ALREADY_SENT, file=sys.stderr)
    return redact(text, findings)


def _build_pipeline(
    settings: Settings, store: CorpusStore
) -> tuple[EvidencePipeline, int]:
    retriever = Retriever.from_store(store)
    generator = build_llm(settings, role="generator")
    verifier = build_llm(settings, role="verifier") or generator
    config = PipelineConfig(
        top_k=settings.top_k,
        per_query_k=settings.per_query_k,
        min_coverage=settings.min_coverage,
        keep_partial=settings.keep_partial_claims,
    )
    return EvidencePipeline(retriever, generator, verifier, config), len(retriever)


def cmd_status(args: argparse.Namespace, settings: Settings) -> int:
    print(f"KARE {__version__}")
    print(f"Modelo: {_model_label(settings)}")
    if settings.verifier_model:
        print(f"Verificador: {settings.verifier_model}")
    print(
        "Transcripción: "
        f"ElevenLabs {_yes(bool(settings.elevenlabs_api_key))} · "
        f"Whisper {_yes(bool(settings.openai_api_key))}"
    )
    manifest = load_manifest(settings.manifest_path)
    with CorpusStore(settings.db_path) as store:
        counts = store.chunk_counts()
    print(f"\nCorpus ({settings.db_path}):")
    for spec in manifest.sources:
        indexed = counts.get(spec.id, 0)
        print(
            f"  {spec.id:<34} {spec.license_status.value:<13} {indexed:>6} fragmentos"
        )
    orphans = sorted(set(counts) - {spec.id for spec in manifest.sources})
    for source_id in orphans:
        print(f"  {source_id:<34} {'fuera del manifiesto':<13} {counts[source_id]:>6}")
    print(f"Total: {sum(counts.values())} fragmentos indexados.")
    return 0


def cmd_ingest(args: argparse.Namespace, settings: Settings) -> int:
    manifest = load_manifest(settings.manifest_path)
    config = ChunkingConfig(
        target_chars=settings.chunk_target_chars,
        max_chars=settings.chunk_max_chars,
        overlap_chars=settings.chunk_overlap_chars,
    )
    with CorpusStore(settings.db_path) as store:
        reports = ingest_manifest(
            manifest,
            corpus_dir=Path(settings.corpus_dir),
            store=store,
            config=config,
            only=args.solo or None,
        )
    for report in reports:
        print(f"{report.status:<9} {report.source_id}: {report.message}")
    return 1 if any(report.status == "error" for report in reports) else 0


def cmd_ask(args: argparse.Namespace, settings: Settings) -> int:
    question = " ".join(args.pregunta).strip()
    if args.audio:
        transcriber = get_transcriber(settings)
        if transcriber is None:
            print(
                NO_TRANSCRIBER,
                file=sys.stderr,
            )
            return 2
        heard = asyncio.run(transcribe_safely(transcriber, args.audio))
        if not heard:
            print("No se pudo transcribir el audio.", file=sys.stderr)
            return 1
        print(f"Escuché: {_shown_transcript(heard)}\n")
        question = f"{question} {heard}".strip()
    if not question:
        print("Escribe una pregunta o usa --audio.", file=sys.stderr)
        return 2
    with CorpusStore(settings.db_path) as store:
        pipeline, indexed = _build_pipeline(settings, store)
    if indexed == 0:
        print("Aviso: el corpus está vacío; corre `kare ingerir`.\n", file=sys.stderr)
    answer = asyncio.run(pipeline.answer(question))
    if args.json:
        print(answer.model_dump_json(indent=2))
    else:
        print(render_text(answer, show_discarded=args.descartadas))
    return 0


def cmd_scan(args: argparse.Namespace, settings: Settings) -> int:
    if args.archivo:
        text = Path(args.archivo).read_text(encoding="utf-8")
    else:
        text = " ".join(args.texto)
    findings = scan(text)
    if not findings:
        print(
            "Sin identificadores detectados. La guardia no es infalible: revisa igual."
        )
        return 0
    print(describe(findings))
    print()
    print(redact(text, findings))
    return 1


def cmd_transcribe(args: argparse.Namespace, settings: Settings) -> int:
    transcriber = get_transcriber(settings)
    if transcriber is None:
        print(
            NO_TRANSCRIBER,
            file=sys.stderr,
        )
        return 2
    text = asyncio.run(transcribe_safely(transcriber, args.audio))
    if text is None:
        return 1
    print(_shown_transcript(text))
    return 0


def cmd_eval_run(args: argparse.Namespace, settings: Settings) -> int:
    try:
        bank = load_bank(Path(args.banco))
    except BankError as exc:
        print(f"El banco tiene problemas:\n{exc}", file=sys.stderr)
        return 2
    generator = build_llm(settings, role="generator")
    if generator is None:
        print("La evaluación necesita un modelo configurado.", file=sys.stderr)
        return 2
    with CorpusStore(settings.db_path) as store:
        pipeline, indexed = _build_pipeline(settings, store)
        sources = sorted(store.sources())
    if indexed == 0:
        print("El corpus está vacío: corre `kare ingerir` primero.", file=sys.stderr)
        return 2
    metadata: dict[str, object] = {
        "modelo": generator.model_name,
        "verificador": settings.verifier_model or generator.model_name,
        "fuentes": sources,
        "fragmentos": indexed,
        "banco": str(args.banco),
        "configuracion": {
            "top_k": settings.top_k,
            "min_coverage": settings.min_coverage,
            "keep_partial_claims": settings.keep_partial_claims,
        },
    }

    def progress(question_id: str) -> None:
        print(f"  lista {question_id}", file=sys.stderr)

    outcome = asyncio.run(
        run_evaluation(
            bank,
            pipeline,
            BaselineAnswerer(generator),
            Path(args.salida),
            seed=args.semilla,
            concurrency=args.concurrencia,
            metadata=metadata,
            on_progress=progress,
        )
    )
    print("Corrida lista:")
    for path in outcome.paths:
        print(f"  {path}")
    failed = sorted({qid for ids in outcome.failed.values() for qid in ids})
    if failed:
        print(
            f"Aviso: {len(failed)} preguntas fallaron por error del modelo "
            f"({', '.join(failed)}). Vuelve a correrlas en otra carpeta antes de "
            "calificar; el puntaje las excluye.",
            file=sys.stderr,
        )
    print(
        "Comparte solo ciego.csv (una copia por calificadora). Guarda clave.csv "
        "aparte; citas.csv se califica después."
    )
    return 0


def cmd_eval_score(args: argparse.Namespace, settings: Settings) -> int:
    report = score_run(
        Path(args.corrida),
        [Path(path) for path in args.calificaciones],
        [Path(path) for path in args.citas],
    )
    text = render_report(report)
    print(text)
    if args.informe:
        Path(args.informe).write_text(text + "\n", encoding="utf-8")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kare",
        description="KARE-E: evidencia citada para cuidados paliativos oncológicos.",
    )
    parser.add_argument("--version", action="version", version=f"kare {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    status = commands.add_parser("estado", help="configuración y estado del corpus")
    status.set_defaults(handler=cmd_status)

    ingest = commands.add_parser("ingerir", help="indexa las fuentes autorizadas")
    ingest.add_argument("--solo", nargs="+", metavar="ID", help="solo estas fuentes")
    ingest.set_defaults(handler=cmd_ingest)

    ask = commands.add_parser("preguntar", help="responde con evidencia citada")
    ask.add_argument("pregunta", nargs="*", help="la pregunta, sin datos de pacientes")
    ask.add_argument("--audio", help="nota de voz con la pregunta")
    ask.add_argument("--json", action="store_true", help="salida completa en JSON")
    ask.add_argument(
        "--descartadas",
        action="store_true",
        help="muestra lo que se descartó y por qué",
    )
    ask.set_defaults(handler=cmd_ask)

    scan_cmd = commands.add_parser("escanear", help="busca identificadores en un texto")
    scan_cmd.add_argument("texto", nargs="*")
    scan_cmd.add_argument("--archivo", help="archivo de texto a revisar")
    scan_cmd.set_defaults(handler=cmd_scan)

    transcribe = commands.add_parser("transcribir", help="transcribe una nota de voz")
    transcribe.add_argument("audio")
    transcribe.set_defaults(handler=cmd_transcribe)

    evaluate = commands.add_parser(
        "evaluar", help="evaluación ciega frente al modelo solo"
    )
    eval_commands = evaluate.add_subparsers(dest="eval_command", required=True)

    run = eval_commands.add_parser("correr", help="genera la hoja ciega y la clave")
    run.add_argument("--banco", required=True, help="CSV del banco de preguntas")
    run.add_argument("--salida", required=True, help="carpeta nueva para la corrida")
    run.add_argument("--semilla", type=int, default=13)
    run.add_argument("--concurrencia", type=int, default=3)
    run.set_defaults(handler=cmd_eval_run)

    score = eval_commands.add_parser("puntuar", help="calcula las métricas")
    score.add_argument("--corrida", required=True, help="carpeta de la corrida")
    score.add_argument(
        "--calificaciones", nargs="+", required=True, help="hojas ciegas calificadas"
    )
    score.add_argument(
        "--citas", nargs="*", default=[], help="hojas de citas calificadas"
    )
    score.add_argument("--informe", help="guarda el informe en Markdown")
    score.set_defaults(handler=cmd_eval_score)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(), format="%(levelname)s %(name)s: %(message)s"
    )
    handler: Handler = args.handler
    try:
        return handler(args, settings)
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
