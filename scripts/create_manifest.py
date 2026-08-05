from pipeline.manifest import create_manifest


def main():

    summary = create_manifest()

    print("\n========== DATASET MANIFEST ==========\n")

    print(f"Total Videos : {summary['total_videos']}\n")

    for dataset, count in summary["datasets"].items():

        print(f"{dataset:<15} : {count}")

    print("\nManifest saved successfully.")

    print(f"CSV  : {summary['csv']}")
    print(f"JSON : {summary['json']}")


if __name__ == "__main__":
    main()