import torch
import torch.nn as nn


class ReactNet(nn.Module):
    def __init__(self, backbone):
        super(ReactNet, self).__init__()
        self.backbone = backbone

    def forward(
        self,
        x,
        return_feature=False,
        return_feature_list=False,
        preembedded=False,
    ):
        try:
            return self.backbone(
                x,
                return_feature=return_feature,
                return_feature_list=return_feature_list,
                preembedded=preembedded,
            )
        except TypeError:
            try:
                return self.backbone(
                    x,
                    return_feature=return_feature,
                    return_feature_list=return_feature_list,
                )
            except TypeError:
                return self.backbone(x, return_feature)

    def forward_threshold(self, x, threshold, preembedded=False):
        try:
            _, feature = self.backbone(
                x, return_feature=True, preembedded=preembedded
            )
        except TypeError:
            _, feature = self.backbone(x, return_feature=True)
        feature = feature.clip(max=threshold)
        feature = feature.view(feature.size(0), -1)
        logits_cls = self._apply_classifier(feature)
        return logits_cls

    def get_fc(self):
        if hasattr(self.backbone, "get_fc"):
            return self.backbone.get_fc()

        fc_layer = self._get_fc_layer()
        return (
            fc_layer.weight.cpu().detach().numpy(),
            fc_layer.bias.cpu().detach().numpy(),
        )

    def _apply_classifier(self, feature):
        fc_layer = self._get_fc_layer()
        if fc_layer is not None:
            return fc_layer(feature)

        if hasattr(self.backbone, "get_fc"):
            weight, bias = self.backbone.get_fc()
            if not torch.is_tensor(weight):
                weight = torch.as_tensor(
                    weight, device=feature.device, dtype=feature.dtype
                )
            else:
                weight = weight.to(device=feature.device, dtype=feature.dtype)

            if bias is not None:
                if not torch.is_tensor(bias):
                    bias = torch.as_tensor(
                        bias, device=feature.device, dtype=feature.dtype
                    )
                else:
                    bias = bias.to(device=feature.device, dtype=feature.dtype)

            return torch.nn.functional.linear(feature, weight, bias)

        raise AttributeError(
            f"{self.backbone.__class__.__name__} does not expose a classifier "
            "layer (expected one of get_fc_layer/get_fc/fc/classifier/head)."
        )

    def _get_fc_layer(self):
        if hasattr(self.backbone, "get_fc_layer"):
            fc_layer = self.backbone.get_fc_layer()
            if fc_layer is not None:
                return fc_layer

        for attr in ("fc", "classifier", "head"):
            if not hasattr(self.backbone, attr):
                continue
            layer = getattr(self.backbone, attr)
            if isinstance(layer, nn.Linear):
                return layer
            if isinstance(layer, nn.Sequential):
                for sublayer in reversed(layer):
                    if isinstance(sublayer, nn.Linear):
                        return sublayer

        return None
