def quality_check(data):
    # Perform basic quality checks
    null_counts = data.isnull().sum()
    duplicates = data.duplicated().sum()

    quality_report = {
        "total_rows": len(data),
        "null_counts": null_counts.to_dict(),
        "duplicates": duplicates,
    }
    print(quality_report)

    # Annotate quality in the dataset
    # data["quality_flag"] = data.apply(
    #     lambda row: "Incomplete" if row.isnull().any() else "Complete", axis=1
    # )

    # return data, quality_report
    return data, []
