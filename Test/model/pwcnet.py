#!/usr/bin/env python

import math
import torch
import typing

import model.backwarp as backwarp
import model.costvol as costvol


##########################################################


class Basic(torch.nn.Module):
    def __init__(self, strType:str, intChans:typing.List[int], objScratch:typing.Optional[typing.Dict]=None):
        super().__init__()

        self.strType = strType
        self.netEvenize = None
        self.netMain = None
        self.netShortcut = None

        intIn = intChans[0]
        intOut = intChans[-1]
        netMain = []
        intChans = intChans.copy()
        fltStride = 1.0

        for intPart, strPart in enumerate(self.strType.split('+')[0].split('-')):
            if strPart.startswith('evenize') == True and intPart == 0:
                class Evenize(torch.nn.Module):
                    def __init__(self, strPad):
                        super().__init__()

                        self.strPad = strPad
                    # end

                    def forward(self, tenIn:torch.Tensor) -> torch.Tensor:
                        intPad = [0, 0, 0, 0]

                        if tenIn.shape[3] % 2 != 0: intPad[1] = 1
                        if tenIn.shape[2] % 2 != 0: intPad[3] = 1

                        if min(intPad) != 0 or max(intPad) != 0:
                            tenIn = torch.nn.functional.pad(input=tenIn, pad=intPad, mode=self.strPad if self.strPad != 'zeros' else 'constant', value=0.0)
                        # end

                        return tenIn
                    # end
                # end

                strPad = 'zeros'

                if '(' in strPart:
                    if 'replpad' in strPart.split('(')[1].split(')')[0].split(','): strPad = 'replicate'
                    if 'reflpad' in strPart.split('(')[1].split(')')[0].split(','): strPad = 'reflect'
                # end

                self.netEvenize = Evenize(strPad)

            elif strPart.startswith('conv') == True:
                intKsize = 3
                intPad = 1
                strPad = 'zeros'

                if '(' in strPart:
                    intKsize = int(strPart.split('(')[1].split(')')[0].split(',')[0])
                    intPad = int(math.floor(0.5 * (intKsize - 1)))

                    if 'replpad' in strPart.split('(')[1].split(')')[0].split(','): strPad = 'replicate'
                    if 'reflpad' in strPart.split('(')[1].split(')')[0].split(','): strPad = 'reflect'
                # end

                if 'nopad' in self.strType.split('+'):
                    intPad = 0
                # end

                netMain += [torch.nn.Conv2d(in_channels=intChans[0], out_channels=intChans[1], kernel_size=intKsize, stride=1, padding=intPad, padding_mode=strPad, bias='nobias' not in self.strType.split('+'))]
                intChans = intChans[1:]
                fltStride *= 1.0

            elif strPart.startswith('sconv') == True:
                intKsize = 3
                intPad = 1
                strPad = 'zeros'

                if '(' in strPart:
                    intKsize = int(strPart.split('(')[1].split(')')[0].split(',')[0])
                    intPad = int(math.floor(0.5 * (intKsize - 1)))

                    if 'replpad' in strPart.split('(')[1].split(')')[0].split(','): strPad = 'replicate'
                    if 'reflpad' in strPart.split('(')[1].split(')')[0].split(','): strPad = 'reflect'
                # end

                if 'nopad' in self.strType.split('+'):
                    intPad = 0
                # end

                netMain += [torch.nn.Conv2d(in_channels=intChans[0], out_channels=intChans[1], kernel_size=intKsize, stride=2, padding=intPad, padding_mode=strPad, bias='nobias' not in self.strType.split('+'))]
                intChans = intChans[1:]
                fltStride *= 2.0

            elif strPart.startswith('up') == True:
                class Up(torch.nn.Module):
                    def __init__(self, strType):
                        super().__init__()

                        self.strType = strType
                    # end

                    def forward(self, tenIn:torch.Tensor) -> torch.Tensor:
                        if self.strType == 'nearest':
                            return torch.nn.functional.interpolate(input=tenIn, scale_factor=2.0, mode='nearest-exact', align_corners=False)

                        elif self.strType == 'bilinear':
                            return torch.nn.functional.interpolate(input=tenIn, scale_factor=2.0, mode='bilinear', align_corners=False)

                        elif self.strType == 'pyramid':
                            return pyramid(tenIn, None, 'up')

                        elif self.strType == 'shuffle':
                            return torch.nn.functional.pixel_shuffle(tenIn, upscale_factor=2) # https://github.com/pytorch/pytorch/issues/62854

                        # end

                        assert(False) # to make torchscript happy
                    # end
                # end

                strType = 'bilinear'

                if '(' in strPart:
                    if 'nearest' in strPart.split('(')[1].split(')')[0].split(','): strType = 'nearest'
                    if 'pyramid' in strPart.split('(')[1].split(')')[0].split(','): strType = 'pyramid'
                    if 'shuffle' in strPart.split('(')[1].split(')')[0].split(','): strType = 'shuffle'
                # end

                netMain += [Up(strType)]
                fltStride *= 0.5

            elif strPart.startswith('prelu') == True:
                netMain += [torch.nn.PReLU(num_parameters=1, init=float(strPart.split('(')[1].split(')')[0].split(',')[0]))]
                fltStride *= 1.0

            elif True:
                assert(False)

            # end
        # end

        self.netMain = torch.nn.Sequential(*netMain)

        for strPart in self.strType.split('+')[1:]:
            if strPart.startswith('skip') == True:
                if intIn == intOut and fltStride == 1.0:
                    self.netShortcut = torch.nn.Identity()

                elif intIn != intOut and fltStride == 1.0:
                    self.netShortcut = torch.nn.Conv2d(in_channels=intIn, out_channels=intOut, kernel_size=1, stride=1, padding=0, bias='nobias' not in self.strType.split('+'))

                elif intIn == intOut and fltStride != 1.0:
                    class Down(torch.nn.Module):
                        def __init__(self, fltScale):
                            super().__init__()

                            self.fltScale = fltScale
                        # end

                        def forward(self, tenIn:torch.Tensor) -> torch.Tensor:
                            return torch.nn.functional.interpolate(input=tenIn, scale_factor=self.fltScale, mode='bilinear', align_corners=False)
                        # end
                    # end

                    self.netShortcut = Down(1.0 / fltStride)

                elif intIn != intOut and fltStride != 1.0:
                    class Down(torch.nn.Module):
                        def __init__(self, fltScale):
                            super().__init__()

                            self.fltScale = fltScale
                        # end

                        def forward(self, tenIn:torch.Tensor) -> torch.Tensor:
                            return torch.nn.functional.interpolate(input=tenIn, scale_factor=self.fltScale, mode='bilinear', align_corners=False)
                        # end
                    # end

                    self.netShortcut = torch.nn.Sequential(Down(1.0 / fltStride), torch.nn.Conv2d(in_channels=intIn, out_channels=intOut, kernel_size=1, stride=1, padding=0, bias='nobias' not in self.strType.split('+')))

                # end

            elif strPart.startswith('...') == True:
                pass

            # end
        # end

        assert(len(intChans) == 1)
    # end

    def forward(self, tenIn:torch.Tensor) -> torch.Tensor:
        if self.netEvenize is not None:
            tenIn = self.netEvenize(tenIn)
        # end

        tenOut = self.netMain(tenIn)

        if self.netShortcut is not None:
            tenOut = tenOut + self.netShortcut(tenIn)
        # end

        return tenOut
    # end
# end


##########################################################


class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()

        class Extractor(torch.nn.Module):
            def __init__(self):
                super().__init__()

                self.netOne = Basic('evenize(replpad)-sconv(2)-prelu(0.25)-conv(3,replpad)-prelu(0.25)-conv(3,replpad)-prelu(0.25)', [3, 32, 32, 32], None)
                self.netTwo = Basic('evenize(replpad)-sconv(2)-prelu(0.25)-conv(3,replpad)-prelu(0.25)-conv(3,replpad)-prelu(0.25)', [32, 32, 32, 32], None)
                self.netThr = Basic('evenize(replpad)-sconv(2)-prelu(0.25)-conv(3,replpad)-prelu(0.25)-conv(3,replpad)-prelu(0.25)', [32, 32, 32, 32], None)
            # end

            def forward(self, tenIn):
                tenOne = self.netOne(tenIn)
                tenTwo = self.netTwo(tenOne)
                tenThr = self.netThr(tenTwo)
                tenFou = torch.nn.functional.avg_pool2d(input=tenThr, kernel_size=2, stride=2, count_include_pad=False)
                tenFiv = torch.nn.functional.avg_pool2d(input=tenFou, kernel_size=2, stride=2, count_include_pad=False)

                return [tenOne, tenTwo, tenThr, tenFou, tenFiv]
            # end
        # end

        class Decoder(torch.nn.Module):
            def __init__(self, intChannels):
                super().__init__()

                self.netCostacti = torch.nn.PReLU(num_parameters=1, init=0.25)
                self.netMain = Basic('conv(3,replpad)-prelu(0.25)-conv(3,replpad)-prelu(0.25)-conv(3,replpad)-prelu(0.25)-conv(3,replpad)-prelu(0.25)-conv(3,replpad)-prelu(0.25)-conv(3,replpad)', [intChannels, 128, 128, 96, 64, 32, 2], None)
            # end

            def forward(self, tenOne, tenTwo, tenFlow):
                if tenFlow is not None:
                    tenFlow = 2.0 * torch.nn.functional.interpolate(input=tenFlow, scale_factor=2.0, mode='bilinear', align_corners=False)
                # end

                tenMain = []

                if tenFlow is None:
                    tenMain.append(tenOne)
                    tenMain.append(self.netCostacti(costvol.costvol_func.apply(tenOne, tenTwo)))

                elif tenFlow is not None:
                    tenMain.append(tenOne)
                    tenMain.append(self.netCostacti(costvol.costvol_func.apply(tenOne, backwarp.backwarp(tenTwo, tenFlow.detach()))))
                    tenMain.append(tenFlow)

                # end

                return (tenFlow if tenFlow is not None else 0.0) + self.netMain(torch.cat(tenMain, 1))
            # end
        # end

        self.netExtractor = Extractor()

        self.netFiv = Decoder(32 + 81 + 0)
        self.netFou = Decoder(32 + 81 + 2)
        self.netThr = Decoder(32 + 81 + 2)
        self.netTwo = Decoder(32 + 81 + 2)
        self.netOne = Decoder(32 + 81 + 2)
    # end

    def bidir(self, tenOne, tenTwo):
        intWidth = tenOne.shape[3] and tenTwo.shape[3]
        intHeight = tenOne.shape[2] and tenTwo.shape[2]

        tenOne, tenTwo = list(zip(*[torch.split(tenFeat, [tenOne.shape[0], tenTwo.shape[0]], 0) for tenFeat in self.netExtractor(torch.cat([tenOne, tenTwo], 0))]))

        tenFwd = None
        tenFwd = self.netFiv(tenOne[-1], tenTwo[-1], tenFwd)
        tenFwd = self.netFou(tenOne[-2], tenTwo[-2], tenFwd)
        tenFwd = self.netThr(tenOne[-3], tenTwo[-3], tenFwd)
        tenFwd = self.netTwo(tenOne[-4], tenTwo[-4], tenFwd)
        tenFwd = self.netOne(tenOne[-5], tenTwo[-5], tenFwd)

        tenBwd = None
        tenBwd = self.netFiv(tenTwo[-1], tenOne[-1], tenBwd)
        tenBwd = self.netFou(tenTwo[-2], tenOne[-2], tenBwd)
        tenBwd = self.netThr(tenTwo[-3], tenOne[-3], tenBwd)
        tenBwd = self.netTwo(tenTwo[-4], tenOne[-4], tenBwd)
        tenBwd = self.netOne(tenTwo[-5], tenOne[-5], tenBwd)

        return tenFwd, tenBwd
    # end
# end
