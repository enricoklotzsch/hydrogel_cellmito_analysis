from __future__ import annotations

import argparse
import json

from .pipeline import run_directory_pipeline, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="UMAP analysis for flow data")
    parser.add_argument("--input", default=None, help="Path to input .csv/.tsv/.fcs file")
    parser.add_argument("--input-dir", default=None, help="Directory containing .fcs files")
    parser.add_argument("--out-dir", default="flow_umap_analysis/results", help="Output directory")
    parser.add_argument("--channels", nargs="*", default=None, help="Optional channel list")
    parser.add_argument("--sample-size", type=int, default=None, help="Optional random sample size")
    parser.add_argument(
        "--sample-size-per-file",
        type=int,
        default=1500,
        help="Optional per-file random sample size (directory mode)",
    )
    parser.add_argument("--cofactor", type=float, default=150.0, help="Cofactor for arcsinh transform")
    parser.add_argument("--no-arcsinh", action="store_true", help="Disable arcsinh transform")
    parser.add_argument("--n-neighbors", type=int, default=30, help="UMAP n_neighbors")
    parser.add_argument("--min-dist", type=float, default=0.1, help="UMAP min_dist")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--dbscan-eps", type=float, default=0.7, help="DBSCAN eps")
    parser.add_argument("--dbscan-min-samples", type=int, default=20, help="DBSCAN min_samples")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if bool(args.input) == bool(args.input_dir):
        raise ValueError("Provide exactly one of --input or --input-dir")

    if args.input_dir:
        metadata = run_directory_pipeline(
            input_dir=args.input_dir,
            out_dir=args.out_dir,
            channels=args.channels,
            sample_size_per_file=args.sample_size_per_file,
            cofactor=args.cofactor,
            use_arcsinh=not args.no_arcsinh,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            random_state=args.random_state,
            dbscan_eps=args.dbscan_eps,
            dbscan_min_samples=args.dbscan_min_samples,
        )
        print(json.dumps(metadata, indent=2))
        return

    metadata = run_pipeline(
        input_path=args.input,
        out_dir=args.out_dir,
        channels=args.channels,
        sample_size=args.sample_size,
        cofactor=args.cofactor,
        use_arcsinh=not args.no_arcsinh,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        random_state=args.random_state,
        dbscan_eps=args.dbscan_eps,
        dbscan_min_samples=args.dbscan_min_samples,
    )

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
