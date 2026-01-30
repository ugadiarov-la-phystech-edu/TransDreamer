import torch
import torchvision
from omegaconf import OmegaConf

from ocr.dinov2saur.modules.encoders import TimmExtractor, FrameEncoder
from ocr.dinov2saur.modules.groupers import SlotAttention
from ocr.dinov2saur.modules.initializers import RandomInit, FixedLearnedInit
from ocr.dinov2saur.modules.networks import build
from ocr.dinov2saur.modules.video import LatentProcessor
from ocr.tools import SlotExtractor


class DinoV2saur(torch.nn.Module, SlotExtractor):
    def __init__(self, config_path, checkpoint_path, image_size, device):
        super().__init__()
        self._device = device
        self._config_path = config_path
        self._checkpoint_path = checkpoint_path
        self._image_size = image_size
        self._config = OmegaConf.load(self._config_path).model
        self._normalization = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        initializer_kwargs = dict(self._config.initializer)
        initializer_name = initializer_kwargs.pop("name")
        self.initializer = self._instantiate(initializer_name, [RandomInit, FixedLearnedInit], initializer_kwargs)

        backbone_kwargs = dict(self._config.encoder.backbone)
        backbone_name = backbone_kwargs.pop("name")
        backbone = self._instantiate(backbone_name, [TimmExtractor], backbone_kwargs)

        output_transform_config = self._config.encoder.output_transform
        output_transform = build(output_transform_config, 'two_layer_mlp')

        encoder_config = self._config.encoder
        self.encoder = FrameEncoder(backbone=backbone, pos_embed=None, output_transform=output_transform,
                                     spatial_flatten=False, main_features_key=encoder_config.get("main_features_key", "vit_block12"))

        grouper_config = self._config.grouper
        grouper = SlotAttention(inp_dim=grouper_config.inp_dim, slot_dim=grouper_config.slot_dim,
                                     n_iters=grouper_config.n_iters, use_mlp=grouper_config.use_mlp, )
        self.processor = LatentProcessor(grouper, predictor=None)

        state_dict = torch.load(self._checkpoint_path, weights_only=False, map_location='cpu')['state_dict']
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        assert len(missing_keys) == 0, f'{missing_keys}'
        assert all(key.startswith('decoder') for key in unexpected_keys)

        self.to(self._device)
        self.requires_grad_(False)
        self.eval()

    @property
    def n_slots(self):
        return self._config.initializer.n_slots

    @property
    def dim(self):
        return self._config.initializer.dim

    @staticmethod
    def _instantiate(class_name, classes, kwargs):
        for cls in classes:
            if cls.__name__ == class_name:
                return cls(**kwargs)

        raise ValueError(f'Unknown class name: {class_name}')

    def forward(self, image, previous_slots=None):
        encoder_input = self._normalization(image)
        batch_size = image.size()[0]

        encoder_output = self.encoder(encoder_input)
        features = encoder_output["features"]

        slots_initial = previous_slots
        if slots_initial is None:
            slots_initial = self.initializer(batch_size=batch_size)

        processor_output = self.processor(slots_initial, features)
        return processor_output["state"]
