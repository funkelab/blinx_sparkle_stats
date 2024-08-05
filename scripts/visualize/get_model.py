from sparkle_stats.models import (
    Attention,
    GiganticMLP,
    ResNet1D,
    SimpleFullyConnected,
    SimpleVgg,
    Vgg1D,
)


def get_model(architecture, ds, output_classes=None):
    output_classes = output_classes if output_classes is not None else ds.output_classes
    match architecture:
        case "VGG":
            model = Vgg1D(ds.trace_length, output_classes=output_classes)
        case "RESNET":
            model = ResNet1D(output_classes=output_classes, start_channels=32)
        case "SIMPLEFC":
            model = SimpleFullyConnected(output_classes=output_classes)
        case "SIMPLEVGG":
            model = SimpleVgg(ds.trace_length, output_classes=output_classes)
        case "GIGANTICMLP":
            model = GiganticMLP(output_classes=output_classes)
        case "ATTENTION":
            model = Attention(output_classes=output_classes, input_size=ds.trace_length)
        case _:
            assert False
    return model
