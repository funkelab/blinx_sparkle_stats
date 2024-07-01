from torch import nn


class Vgg1D(nn.Module):
    def __init__(
        self,
        input_size,
        feature_maps=32,
        downsample_factors=(2, 2, 2, 2),
        output_classes=7,
        input_feature_maps=1,
    ):
        """
        feature_maps: number of convolutional filters
        input_feature_maps: channels in the input
        """

        super().__init__()

        current_feature_maps = input_feature_maps
        current_size = input_size

        features = []
        for i in range(len(downsample_factors)):
            features += [
                nn.Conv1d(current_feature_maps, feature_maps, kernel_size=3, padding=1),
                nn.BatchNorm1d(feature_maps),
                nn.ReLU(inplace=True),
                nn.Conv1d(feature_maps, feature_maps, kernel_size=3, padding=1),
                nn.BatchNorm1d(feature_maps),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(downsample_factors[i]),
            ]

            current_feature_maps = feature_maps
            feature_maps *= 2

            # check if can downsample
            size = int(current_size / downsample_factors[i])
            check = size * downsample_factors[i] == current_size
            assert (
                check
            ), f"Can not downsample {current_size} by chosen downsample factor"
            current_size = size

        self.features = nn.Sequential(*features)

        # feed forward at the end of the cnn
        classifier = [
            nn.Linear(current_size * current_feature_maps, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, output_classes),
        ]

        self.classifier = nn.Sequential(*classifier)

    def forward(self, raw):
        # pass through cnn
        f = self.features(raw)
        # reshape to match feed forward
        f = f.view(f.size(0), -1)
        y = self.classifier(f)

        return y
