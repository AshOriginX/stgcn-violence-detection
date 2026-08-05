from pipeline.validator import validate_dataset


def main():

    summary = validate_dataset()

    print("\n========== VALIDATION SUMMARY ==========\n")

    print(f"Total Videos   : {summary['total_videos']}")
    print(f"Valid Videos   : {summary['valid_videos']}")
    print(f"Invalid Videos : {summary['invalid_videos']}")

    print("\nPer Dataset\n")

    for dataset, stats in summary["datasets"].items():

        print(
            f"{dataset:<15}"
            f" Total: {stats['total']:<5}"
            f" Valid: {stats['valid']:<5}"
            f" Invalid: {stats['invalid']}"
        )


if __name__ == "__main__":
    main()