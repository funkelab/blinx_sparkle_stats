def expand_name(name):
    return f"full_datasets/{name}_dataset_v6"


def fill_full_dataset(name):
    return "/nrs/funke/projects/blinx/" + expand_name(name)


def get_dataset(model):
    num = int(model.split("_")[0])

    # dataset, normalization dataset
    match num:
        case 3 | 19:
            return (
                expand_name("val"),
                fill_full_dataset("val"),
                fill_full_dataset("train"),
            )
        case 16 | 18:
            return (
                expand_name("val"),
                fill_full_dataset("playground_train"),
                fill_full_dataset("playground_train"),
            )
        case _:
            raise NotImplementedError(f"{num} is not filled in")
