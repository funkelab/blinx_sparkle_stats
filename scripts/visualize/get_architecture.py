def get_architecture(model):
    num = int(model.split("_")[0])

    match num:
        case 3 | 18 | 19:
            return "RESNET"
        case 16:
            return "ATTENTION"
        case _:
            raise NotImplementedError(f"{num} is not filled in")
