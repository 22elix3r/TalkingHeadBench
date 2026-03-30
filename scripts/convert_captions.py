#!/usr/bin/env python3
"""Convert caption JSON entries into clip/image prompt files plus metadata."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

COMPLEX_BACKGROUND_KEYWORDS = [
    "workshop",
    "lab",
    "studio",
    "console",
    "kitchen",
    "library",
    "cafe",
    "office",
    "outdoor",
]

SIMPLE_BACKGROUND_KEYWORDS = ["white", "dark", "plain", "wall", "backdrop"]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Convert captions JSON into prompt text files and metadata."
    )
    parser.add_argument("--input", required=True, type=Path, help="Path to captions.json")
    parser.add_argument(
        "--output-clips", required=True, type=Path, help="Output prompt file for clips"
    )
    parser.add_argument(
        "--output-images",
        required=True,
        type=Path,
        help="Output prompt file for reference images",
    )
    return parser.parse_args()


def load_captions(path: Path) -> list[dict]:
    """Load captions from either a bare list or a wrapped object format."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        print("Detected format: bare list")
        return data
    if isinstance(data, dict):
        for key in ("captions", "data", "items", "entries"):
            if key in data and isinstance(data[key], list):
                print(f"Detected format: wrapped object (key='{key}')")
                return data[key]
        # last resort: find the first list value
        for key, value in data.items():
            if isinstance(value, list):
                print(f"Detected format: wrapped object (key='{key}')")
                return value
    raise ValueError(
        f"Cannot parse captions from {path} - expected a list or "
        f"an object with a 'captions' key. Got: {type(data)}"
    )


def extract_caption_text(entry: Any, index: int) -> str:
    """Extract caption text from an entry with tolerant key handling."""
    if isinstance(entry, str):
        return entry

    if isinstance(entry, dict):
        for key in ("caption", "captions", "text", "description", "prompt"):
            value = entry.get(key)
            if isinstance(value, str):
                return value

    print(
        f"Warning: entry {index} has no readable caption field; using empty caption",
        file=sys.stderr,
    )
    return ""


def extract_original_file(entry: Any) -> str:
    """Extract original filename from an entry with tolerant key handling."""
    if not isinstance(entry, dict):
        return ""

    candidate_keys = (
        "original_file",
        "original_filename",
        "filename",
        "file",
        "video",
        "video_file",
        "video_filename",
        "source_file",
        "path",
        "name",
    )

    for key in candidate_keys:
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return Path(value.strip()).name

    return ""


def split_visual_and_speech(caption_text: str) -> tuple[str, str, bool]:
    """Split caption text into visual and speech parts using [VISUAL]/[SPEECH]."""
    visual_scope = caption_text
    if "[VISUAL]" in visual_scope:
        visual_scope = visual_scope.split("[VISUAL]", 1)[1]

    if "[SPEECH]" in visual_scope:
        visual_part, speech_part = visual_scope.split("[SPEECH]", 1)
        return visual_part.strip(), speech_part.strip(), True

    return visual_scope.strip(), "", False


def extract_transcript(speech_part: str) -> str:
    """Extract transcript from the outermost quote pair (double first, then single)."""
    candidate = speech_part.strip()

    for quote_char in ('"', "'"):
        start = candidate.find(quote_char)
        end = candidate.rfind(quote_char)
        if start != -1 and end != -1 and end > start:
            inner = candidate[start + 1 : end].strip()
            return inner.strip('"\'').strip()

    return candidate.strip('"\'').strip()


def extract_speaker_style(speech_part: str) -> str:
    """Extract style cue between 'speaks' and ':' if present."""
    match = re.search(r"\bspeaks?\s+([^:]+):", speech_part, flags=re.IGNORECASE)
    if not match:
        return ""
    return match.group(1).strip()


def infer_pose_hint(original_file: str) -> str:
    """Infer pose hint from original filename."""
    filename = original_file.lower()

    if any(token in filename for token in ("threequarter", "side", "profile")):
        return "non_frontal"
    if "front" in filename:
        return "frontal"
    return "unknown"


def infer_background_hint(original_file: str) -> str:
    """Infer background hint from original filename."""
    filename = original_file.lower()

    if any(token in filename for token in COMPLEX_BACKGROUND_KEYWORDS):
        return "complex"
    if any(token in filename for token in SIMPLE_BACKGROUND_KEYWORDS):
        return "simple"
    return "unknown"


def write_lines(path: Path, lines: list[str]) -> None:
    """Write text lines with trailing newline if non-empty."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(lines)
    if lines:
        body += "\n"
    path.write_text(body, encoding="utf-8")


def validate_prompt_file(path: Path, expected_count: int) -> list[str]:
    """Validate prompt file line count and 'stem: text' formatting."""
    failures: list[str] = []

    if not path.exists():
        return [f"Missing output file: {path}"]

    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != expected_count:
        failures.append(
            f"{path}: expected {expected_count} lines, found {len(lines)}"
        )

    pattern = re.compile(r"^[^\s:]+: .*$")
    for line_no, line in enumerate(lines, start=1):
        if not pattern.match(line):
            failures.append(
                f"{path}: line {line_no} is not in 'stem: text' format -> {line!r}"
            )
            continue

        stem = line.split(":", 1)[0]
        if any(ch.isspace() for ch in stem):
            failures.append(
                f"{path}: line {line_no} stem contains whitespace -> {stem!r}"
            )

    return failures


def warn_file_count_mismatch(caption_count: int, clips_dir: Path) -> None:
    """Warn when caption count differs from available clip files."""
    if clips_dir.exists() and clips_dir.is_dir():
        clip_files = sorted([p for p in clips_dir.iterdir() if p.is_file()])
    else:
        clip_files = []

    clip_count = len(clip_files)
    if caption_count < clip_count:
        unmatched = ", ".join(p.name for p in clip_files[caption_count:])
        print(
            (
                "Warning: captions.json has fewer entries than files in "
                f"{clips_dir}/ ({caption_count} < {clip_count}). "
                f"Unmatched files: {unmatched}"
            ),
            file=sys.stderr,
        )
    elif caption_count > clip_count:
        missing_stems = ", ".join(
            f"clip_{str(i + 1).zfill(3)}" for i in range(clip_count, caption_count)
        )
        print(
            (
                "Warning: captions.json has more entries than files in "
                f"{clips_dir}/ ({caption_count} > {clip_count}). "
                f"Missing files for: {missing_stems}"
            ),
            file=sys.stderr,
        )


def main() -> int:
    """Run conversion, write outputs, and validate generated files."""
    args = parse_args()
    metadata_path = Path("data/prompts_metadata.json")

    entries = load_captions(Path(args.input))
    warn_file_count_mismatch(len(entries), Path("data/clips"))

    clip_lines: list[str] = []
    image_lines: list[str] = []
    metadata_rows: list[dict[str, Any]] = []

    frontal_stems: list[str] = []
    non_frontal_stems: list[str] = []
    complex_bg_count = 0
    simple_bg_count = 0
    unknown_bg_count = 0

    for i, entry in enumerate(entries):
        padded_num = str(i + 1).zfill(3)
        clip_stem = f"clip_{padded_num}"
        image_stem = f"ref_{padded_num}"

        caption_text = extract_caption_text(entry, i)
        original_file = extract_original_file(entry)

        visual_part, speech_part, has_speech = split_visual_and_speech(caption_text)
        transcript = extract_transcript(speech_part) if has_speech else ""
        speaker_style = extract_speaker_style(speech_part) if has_speech else ""

        if has_speech:
            prompt = (
                f"{visual_part} The person is talking, and he says: '{transcript}'"
            ).strip()
        else:
            prompt = visual_part
            print(
                f"Warning: entry {i} is missing [SPEECH]; using visual-only prompt",
                file=sys.stderr,
            )

        pose_hint = infer_pose_hint(original_file)
        background_hint = infer_background_hint(original_file)

        if pose_hint == "frontal":
            frontal_stems.append(clip_stem)
        elif pose_hint == "non_frontal":
            non_frontal_stems.append(clip_stem)

        if background_hint == "complex":
            complex_bg_count += 1
        elif background_hint == "simple":
            simple_bg_count += 1
        else:
            unknown_bg_count += 1

        clip_lines.append(f"{clip_stem}: {prompt}")
        image_lines.append(f"{image_stem}: {prompt}")

        metadata_rows.append(
            {
                "index": i,
                "original_file": original_file,
                "clip_stem": clip_stem,
                "image_stem": image_stem,
                "visual": visual_part,
                "transcript": transcript,
                "speaker_style": speaker_style,
                "prompt": prompt,
                "pose_hint": pose_hint,
                "background_hint": background_hint,
            }
        )

    write_lines(args.output_clips, clip_lines)
    write_lines(args.output_images, image_lines)

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata_rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    frontal_list = ", ".join(frontal_stems)
    non_frontal_list = ", ".join(non_frontal_stems)
    print(f"Converted {len(entries)} entries")
    print(f"Frontal poses:    {len(frontal_stems)} ({frontal_list})")
    print(f"Non-frontal:      {len(non_frontal_stems)} ({non_frontal_list})")
    print(f"Complex backgrounds: {complex_bg_count}")
    print(f"Simple backgrounds:  {simple_bg_count}")
    print(f"Unknown:          {unknown_bg_count}")
    print()
    print(f"Written: {args.output_clips}")
    print(f"         {args.output_images}")
    print(f"         {metadata_path}")

    validation_failures = []
    validation_failures.extend(validate_prompt_file(args.output_clips, len(entries)))
    validation_failures.extend(validate_prompt_file(args.output_images, len(entries)))

    if validation_failures:
        print("Validation failed:")
        for failure in validation_failures:
            print(f"- {failure}")
        return 1

    print("Validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
