from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn as nn


class GMSR_FP32(nn.Module):
    def __init__(
        self,
        scale=4,
        num_input_channels=12,
        channel=192,
        df_num=7,
    ):
        super(GMSR_FP32, self).__init__()

        self.sf = nn.Conv2d(
            in_channels=num_input_channels,
            out_channels=channel,
            kernel_size=3,
            padding=1,
        )

        self.df = []
        for _ in range(df_num):
            self.df.append(
                nn.Conv2d(
                    in_channels=channel,
                    out_channels=channel,
                    kernel_size=3,
                    padding=1,
                )
            )
            self.df.append(nn.ReLU())
        self.df = nn.Sequential(*self.df)

        self.transition = nn.Conv2d(
            in_channels=channel,
            out_channels=channel,
            kernel_size=3,
            padding=1,
        )

        self.last_conv = nn.Conv2d(
            in_channels=channel + num_input_channels,
            out_channels=3 * (scale**2),
            kernel_size=1,
            padding=0,
        )

    def forward(self, x):
        x = torch.nn.functional.pixel_unshuffle(x, 2)
        img = x

        sf_feat = self.sf(x)

        feat = self.df(sf_feat)

        feat = feat + sf_feat

        feat = self.transition(feat)

        feat = torch.cat([feat, img], dim=1)

        feat = self.last_conv(feat)
        feat = torch.clamp(feat, 0.0, 1.0)
        out = torch.nn.functional.pixel_shuffle(feat, 4)

        return out


if __name__ == "__main__":
    import time

    from fvcore.nn import flop_count_table, FlopCountAnalysis

    model = GMSR_FP32().cuda()
    model.eval()
    inputs = (torch.rand(1, 3, 256, 256).cuda(),)
    print(flop_count_table(FlopCountAnalysis(model, inputs)))

    total_time = 0
    input_x = torch.rand(1, 3, 512, 512).cuda()
    for i in range(100):
        torch.cuda.empty_cache()
        sta_time = time.time()
        model(input_x)
        one_time = time.time() - sta_time
        total_time += one_time * 1000
        print("idx: {} one time: {:.4f} ms".format(i, one_time))
    print("Avg time: {:.4f} ms".format(total_time / 100.0))
