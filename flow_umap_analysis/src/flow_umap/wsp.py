from __future__ import annotations

from pathlib import Path
import re
from typing import Any
from urllib.parse import unquote, urlparse
import xml.etree.ElementTree as ET


def _uri_to_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path))
    return Path(uri)


def _split_tokens(text: str) -> list[str]:
    if not text:
        return []
    return [t for t in re.split(r"[_\s]+", text.strip()) if t]


def _extract_timepoint(text: str) -> str:
    m = re.search(
        r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)\s*(h|hr|hrs|hour|hours|d|day|days|w|wk|week|weeks)(?![A-Za-z0-9])",
        text,
        flags=re.IGNORECASE,
    )
    if not m:
        return ""
    unit = m.group(2).lower()
    if unit in {"hr", "hrs", "hour", "hours"}:
        unit = "h"
    elif unit in {"day", "days"}:
        unit = "d"
    elif unit in {"wk", "week", "weeks"}:
        unit = "w"
    return f"{m.group(1)}{unit}"


def _extract_donor(token: str) -> str:
    direct = re.fullmatch(r"(?i)(donor)?\s*(d\d+)", token)
    if direct:
        return direct.group(2).upper()
    embedded = re.search(r"(?i)(d\d+)", token)
    if embedded:
        return embedded.group(1).upper()
    return ""


def _extract_well(condition_text: str) -> tuple[str, str]:
    m = re.search(r"-([A-H]\d{1,2})$", condition_text, flags=re.IGNORECASE)
    if not m:
        return condition_text, ""
    return condition_text[: m.start()], m.group(1).upper()


def _infer_design_fields(sample_label: str, annotation: str, workspace_name: str) -> dict[str, str]:
    stem = Path(sample_label).stem
    tokens = _split_tokens(stem)

    donor = ""
    stim_state = ""
    condition = ""
    if tokens:
        donor = _extract_donor(tokens[0])
    if len(tokens) > 1:
        stim_state = tokens[1]
    if len(tokens) > 2:
        condition = tokens[2]
    elif tokens:
        condition = tokens[-1]

    condition, well = _extract_well(condition)

    combined_text = " ".join([sample_label, annotation, workspace_name])
    timepoint = _extract_timepoint(combined_text)

    return {
        "sample_id": stem,
        "donor": donor,
        "stim_state": stim_state,
        "condition": condition,
        "timepoint": timepoint,
        "well": well,
    }


def parse_flowjo_workspace(wsp_path: str | Path) -> dict[str, Any]:
    path = Path(wsp_path)
    tree = ET.parse(path)
    root = tree.getroot()

    datasets: dict[str, dict[str, str]] = {}
    for ds in root.findall(".//DataSet"):
        sample_id = ds.attrib.get("sampleID", "")
        uri = ds.attrib.get("uri", "")
        if not sample_id:
            continue
        datasets[sample_id] = {
            "uri": uri,
            "basename": _uri_to_path(uri).name if uri else "",
        }

    sample_nodes: dict[str, dict[str, str]] = {}
    for sn in root.findall(".//SampleNode"):
        sample_id = sn.attrib.get("sampleID", "")
        if not sample_id:
            continue
        sample_nodes[sample_id] = {
            "name": sn.attrib.get("name", ""),
            "annotation": sn.attrib.get("annotation", ""),
        }

    sample_keywords: dict[str, dict[str, str]] = {}
    for sample in root.findall(".//Sample"):
        sample_id = sample.attrib.get("sampleID", "")
        if not sample_id:
            continue
        kws: dict[str, str] = {}
        for kw in sample.findall(".//Keyword"):
            name = kw.attrib.get("name", "")
            value = kw.attrib.get("value", "")
            if name:
                kws[name] = value
        sample_keywords[sample_id] = kws

    groups: dict[str, list[str]] = {}
    for group_node in root.findall(".//GroupNode"):
        gname = group_node.attrib.get("name", "UnnamedGroup")
        refs = [sr.attrib.get("sampleID", "") for sr in group_node.findall(".//SampleRefs/SampleRef")]
        refs = [r for r in refs if r]
        if refs:
            groups[gname] = refs

    workspace_name = path.name
    sample_metadata: dict[str, dict[str, str]] = {}
    for sid, ds in datasets.items():
        basename = ds.get("basename", "")
        node_name = sample_nodes.get(sid, {}).get("name", "")
        annotation = sample_nodes.get(sid, {}).get("annotation", "")
        keywords = sample_keywords.get(sid, {})

        display_name = node_name or basename or keywords.get("$FIL", "")
        inferred = _infer_design_fields(display_name, annotation, workspace_name)

        keyword_timepoint = _extract_timepoint(" ".join([keywords.get("$FIL", ""), keywords.get("TBNM", "")]))
        if keyword_timepoint and not inferred["timepoint"]:
            inferred["timepoint"] = keyword_timepoint

        sample_metadata[sid] = {
            **inferred,
            "sample_name": display_name,
            "annotation": annotation,
            "file_basename": basename,
            "file_stem": Path(basename).stem if basename else "",
        }

    return {
        "workspace_path": str(path),
        "datasets": datasets,
        "sample_nodes": sample_nodes,
        "sample_metadata": sample_metadata,
        "groups": groups,
    }


def workspace_sample_metadata_map(
    wsp_info: dict[str, Any],
    group_name: str | None = None,
) -> dict[str, dict[str, str]]:
    if group_name and group_name in wsp_info.get("groups", {}):
        sample_ids = wsp_info["groups"][group_name]
    else:
        sample_ids = sorted(wsp_info.get("datasets", {}).keys(), key=lambda x: int(x) if x.isdigit() else x)

    out: dict[str, dict[str, str]] = {}
    for sid in sample_ids:
        md = wsp_info.get("sample_metadata", {}).get(sid, {})
        basename = md.get("file_basename", "")
        stem = md.get("file_stem", "")
        if basename:
            out[basename] = md
        if stem:
            out[stem] = md
    return out


def resolve_workspace_files(
    wsp_info: dict[str, Any],
    fcs_dir: str | Path,
    group_name: str | None = None,
) -> list[Path]:
    base_dir = Path(fcs_dir)

    if group_name and group_name in wsp_info["groups"]:
        sample_ids = wsp_info["groups"][group_name]
    else:
        sample_ids = sorted(wsp_info["datasets"].keys(), key=lambda x: int(x) if x.isdigit() else x)

    basename_index: dict[str, list[Path]] = {}

    def _candidates_for_basename(basename: str) -> list[Path]:
        if basename in basename_index:
            return basename_index[basename]
        basename_index[basename] = list(base_dir.rglob(basename))
        return basename_index[basename]

    files: list[Path] = []
    for sid in sample_ids:
        ds = wsp_info["datasets"].get(sid, {})
        basename = ds.get("basename") or wsp_info["sample_nodes"].get(sid, {}).get("name", "")
        if not basename:
            continue
        candidate = base_dir / basename
        if candidate.exists():
            files.append(candidate)
            continue

        uri = ds.get("uri", "")
        uri_parent_name = _uri_to_path(uri).parent.name if uri else ""
        candidates = _candidates_for_basename(basename)
        if not candidates:
            continue

        if uri_parent_name:
            same_parent = [p for p in candidates if p.parent.name == uri_parent_name]
            if same_parent:
                files.append(same_parent[0])
                continue

        files.append(candidates[0])

    # Deduplicate while preserving order.
    seen = set()
    uniq: list[Path] = []
    for f in files:
        if f not in seen:
            uniq.append(f)
            seen.add(f)
    return uniq
