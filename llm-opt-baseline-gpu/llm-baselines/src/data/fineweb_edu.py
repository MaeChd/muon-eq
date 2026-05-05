def get_fineweb_edu_data(datasets_dir, num_proc=40):
    del num_proc

    from .fineweb import _resolve_local_bin_dataset

    return _resolve_local_bin_dataset(
        datasets_dir,
        dataset_name="finewebedu",
        candidate_subdirs=["fineweb-edu-100BT", "fineweb-edu", "finewebedu"],
    )


if __name__ == "__main__":
    get_fineweb_edu_data("./datasets/")
