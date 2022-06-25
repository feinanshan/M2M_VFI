#!/usr/bin/env python

import collections
import cupy
import os
import re
import torch
import typing


##########################################################


objCudacache = {}


def cuda_int32(intIn:int):
    return cupy.int32(intIn)
# end


def cuda_float32(fltIn:float):
    return cupy.float32(fltIn)
# end


def cuda_kernel(strFunction:str, strKernel:str, objVariables:typing.Dict):
    if 'device' not in objCudacache:
        objCudacache['device'] = torch.cuda.get_device_name()
    # end

    strKey = strFunction

    for strVariable in objVariables:
        objValue = objVariables[strVariable]

        strKey += strVariable

        if objValue is None:
            continue

        elif type(objValue) == int:
            strKey += str(objValue)

        elif type(objValue) == float:
            strKey += str(objValue)

        elif type(objValue) == bool:
            strKey += str(objValue)

        elif type(objValue) == str:
            strKey += objValue

        elif type(objValue) == torch.Tensor:
            strKey += str(objValue.dtype)
            strKey += str(objValue.shape)
            strKey += str(objValue.stride())

        elif True:
            print(strVariable, type(objValue))
            assert(False)

        # end
    # end

    strKey += objCudacache['device']

    if strKey not in objCudacache:
        for strVariable in objVariables:
            objValue = objVariables[strVariable]

            if objValue is None:
                continue

            elif type(objValue) == int:
                strKernel = strKernel.replace('{{' + strVariable + '}}', str(objValue))

            elif type(objValue) == float:
                strKernel = strKernel.replace('{{' + strVariable + '}}', str(objValue))

            elif type(objValue) == bool:
                strKernel = strKernel.replace('{{' + strVariable + '}}', str(objValue))

            elif type(objValue) == str:
                strKernel = strKernel.replace('{{' + strVariable + '}}', objValue)

            elif type(objValue) == torch.Tensor and objValue.dtype == torch.uint8:
                strKernel = strKernel.replace('{{type}}', 'unsigned char')

            elif type(objValue) == torch.Tensor and objValue.dtype == torch.float16:
                strKernel = strKernel.replace('{{type}}', 'half')

            elif type(objValue) == torch.Tensor and objValue.dtype == torch.float32:
                strKernel = strKernel.replace('{{type}}', 'float')

            elif type(objValue) == torch.Tensor and objValue.dtype == torch.float64:
                strKernel = strKernel.replace('{{type}}', 'double')

            elif type(objValue) == torch.Tensor and objValue.dtype == torch.int32:
                strKernel = strKernel.replace('{{type}}', 'int')

            elif type(objValue) == torch.Tensor and objValue.dtype == torch.int64:
                strKernel = strKernel.replace('{{type}}', 'long')

            elif type(objValue) == torch.Tensor:
                print(strVariable, objValue.dtype)
                assert(False)

            elif True:
                print(strVariable, type(objValue))
                assert(False)

            # end
        # end

        while True:
            objMatch = re.search('(SIZE_)([0-4])(\()([^\)]*)(\))', strKernel)

            if objMatch is None:
                break
            # end

            intArg = int(objMatch.group(2))

            strTensor = objMatch.group(4)
            intSizes = objVariables[strTensor].size()

            strKernel = strKernel.replace(objMatch.group(), str(intSizes[intArg] if torch.is_tensor(intSizes[intArg]) == False else intSizes[intArg].item()))
        # end

        while True:
            objMatch = re.search('(OFFSET_)([0-4])(\()', strKernel)

            if objMatch is None:
                break
            # end

            intStart = objMatch.span()[1]
            intStop = objMatch.span()[1]
            intParentheses = 1

            while True:
                intParentheses += 1 if strKernel[intStop] == '(' else 0
                intParentheses -= 1 if strKernel[intStop] == ')' else 0

                if intParentheses == 0:
                    break
                # end

                intStop += 1
            # end

            intArgs = int(objMatch.group(2))
            strArgs = strKernel[intStart:intStop].split(',')

            assert(intArgs == len(strArgs) - 1)

            strTensor = strArgs[0]
            intStrides = objVariables[strTensor].stride()

            strIndex = []

            for intArg in range(intArgs):
                strIndex.append('((' + strArgs[intArg + 1].replace('{', '(').replace('}', ')').strip() + ')*' + str(intStrides[intArg] if torch.is_tensor(intStrides[intArg]) == False else intStrides[intArg].item()) + ')')
            # end

            strKernel = strKernel.replace('OFFSET_' + str(intArgs) + '(' + strKernel[intStart:intStop] + ')', '(' + str.join('+', strIndex) + ')')
        # end

        while True:
            objMatch = re.search('(VALUE_)([0-4])(\()', strKernel)

            if objMatch is None:
                break
            # end

            intStart = objMatch.span()[1]
            intStop = objMatch.span()[1]
            intParentheses = 1

            while True:
                intParentheses += 1 if strKernel[intStop] == '(' else 0
                intParentheses -= 1 if strKernel[intStop] == ')' else 0

                if intParentheses == 0:
                    break
                # end

                intStop += 1
            # end

            intArgs = int(objMatch.group(2))
            strArgs = strKernel[intStart:intStop].split(',')

            assert(intArgs == len(strArgs) - 1)

            strTensor = strArgs[0]
            intStrides = objVariables[strTensor].stride()

            strIndex = []

            for intArg in range(intArgs):
                strIndex.append('((' + strArgs[intArg + 1].replace('{', '(').replace('}', ')').strip() + ')*' + str(intStrides[intArg] if torch.is_tensor(intStrides[intArg]) == False else intStrides[intArg].item()) + ')')
            # end

            strKernel = strKernel.replace('VALUE_' + str(intArgs) + '(' + strKernel[intStart:intStop] + ')', strTensor + '[' + str.join('+', strIndex) + ']')
        # end

        objCudacache[strKey] = {
            'strFunction': strFunction,
            'strKernel': strKernel
        }
    # end

    return strKey
# end


@cupy.memoize(for_each_device=True)
def cuda_launch(strKey:str):
    if 'CUDA_HOME' not in os.environ:
        os.environ['CUDA_HOME'] = '/usr/local/cuda/'
    # end

    return cupy.cuda.compile_with_cache(objCudacache[strKey]['strKernel'], tuple(['-I ' + os.environ['CUDA_HOME'], '-I ' + os.environ['CUDA_HOME'] + '/include'])).get_function(objCudacache[strKey]['strFunction'])
# end


##########################################################


class costvol_func(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(self, tenOne, tenTwo):
        tenOut = tenOne.new_empty([tenOne.shape[0], 81, tenOne.shape[2], tenOne.shape[3]])

        cuda_launch(cuda_kernel('costvol_out', '''
            extern "C" __global__ void __launch_bounds__(512) costvol_out(
                const int n,
                const {{type}}* __restrict__ tenOne,
                const {{type}}* __restrict__ tenTwo,
                {{type}}* __restrict__ tenOut
            ) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
                const int intN = ( intIndex / SIZE_3(tenOut) / SIZE_2(tenOut) ) % SIZE_0(tenOut);
                const int intC = -1;
                const int intY = ( intIndex / SIZE_3(tenOut)                  ) % SIZE_2(tenOut);
                const int intX = ( intIndex                                   ) % SIZE_3(tenOut);

                {{type}} fltOne[{{intChans}}];

                for (int intValue = 0; intValue < SIZE_1(tenOne); intValue += 1) {
                    fltOne[intValue] = VALUE_4(tenOne, intN, intValue, intY, intX);
                }

                int intOffset = OFFSET_4(tenOut, intN, 0, intY, intX);

                for (int intOy = intY - 4; intOy <= intY + 4; intOy += 1) {
                    for (int intOx = intX - 4; intOx <= intX + 4; intOx += 1) {
                        {{type}} fltValue = 0.0f;

                        if ((intOy >= 0) && (intOy < SIZE_2(tenOut)) && (intOx >= 0) && (intOx < SIZE_3(tenOut))) {
                            for (int intValue = 0; intValue < SIZE_1(tenOne); intValue += 1) {
                                fltValue += abs(fltOne[intValue] - VALUE_4(tenTwo, intN, intValue, intOy, intOx));
                            }
                        } else {
                            for (int intValue = 0; intValue < SIZE_1(tenOne); intValue += 1) {
                                fltValue += abs(fltOne[intValue]);
                            }
                        }

                        tenOut[intOffset] = fltValue / SIZE_1(tenOne);
                        intOffset += SIZE_2(tenOut) * SIZE_3(tenOut);
                    }
                }
            } }
        ''', {
            'intChans': tenOne.shape[1],
            'tenOne': tenOne,
            'tenTwo': tenTwo,
            'tenOut': tenOut
        }))(
            grid=tuple([int(((tenOut.shape[0] * tenOut.shape[2] * tenOut.shape[3]) + 512 - 1) / 512), 1, 1]),
            block=tuple([512, 1, 1]),
            args=[cuda_int32(tenOut.shape[0] * tenOut.shape[2] * tenOut.shape[3]), tenOne.data_ptr(), tenTwo.data_ptr(), tenOut.data_ptr()],
            stream=collections.namedtuple('Stream', 'ptr')(torch.cuda.current_stream().cuda_stream)
        )

        self.save_for_backward(tenOne, tenTwo)

        return tenOut
    # end

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(self, tenOutgrad):
        tenOne, tenTwo = self.saved_tensors

        tenOutgrad = tenOutgrad.contiguous(); assert(tenOutgrad.is_cuda == True)

        tenOnegrad = tenOne.new_zeros([tenOne.shape[0], tenOne.shape[1], tenOne.shape[2], tenOne.shape[3]]) if self.needs_input_grad[0] == True else None
        tenTwograd = tenTwo.new_zeros([tenTwo.shape[0], tenTwo.shape[1], tenTwo.shape[2], tenTwo.shape[3]]) if self.needs_input_grad[1] == True else None

        if tenOnegrad is not None:
            cuda_launch(cuda_kernel('costvol_onegrad', '''
                extern "C" __global__ void __launch_bounds__(512) costvol_onegrad(
                    const int n,
                    const {{type}}* __restrict__ tenOne,
                    const {{type}}* __restrict__ tenTwo,
                    const {{type}}* __restrict__ tenOutgrad,
                    {{type}}* __restrict__ tenOnegrad,
                    {{type}}* __restrict__ tenTwograd
                ) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
                    const int intN = ( intIndex / SIZE_3(tenOnegrad) / SIZE_2(tenOnegrad) ) % SIZE_0(tenOnegrad);
                    const int intC = -1;
                    const int intY = ( intIndex / SIZE_3(tenOnegrad)                      ) % SIZE_2(tenOnegrad);
                    const int intX = ( intIndex                                           ) % SIZE_3(tenOnegrad);

                    {{type}} fltOne[{{intChans}}];

                    for (int intValue = 0; intValue < SIZE_1(tenOne); intValue += 1) {
                        fltOne[intValue] = VALUE_4(tenOne, intN, intValue, intY, intX);
                    }

                    int intOffset = OFFSET_4(tenOutgrad, intN, 0, intY, intX);

                    for (int intOy = intY - 4; intOy <= intY + 4; intOy += 1) {
                        for (int intOx = intX - 4; intOx <= intX + 4; intOx += 1) {
                            if ((intOy >= 0) && (intOy < SIZE_2(tenOutgrad)) && (intOx >= 0) && (intOx < SIZE_3(tenOutgrad))) {
                                for (int intValue = 0; intValue < SIZE_1(tenOne); intValue += 1) {
                                    if (fltOne[intValue] - VALUE_4(tenTwo, intN, intValue, intOy, intOx) >= 0.0f) {
                                        tenOnegrad[OFFSET_4(tenOnegrad, intN, intValue, intY, intX)] += +tenOutgrad[intOffset] / SIZE_1(tenOne);
                                    } else {
                                        tenOnegrad[OFFSET_4(tenOnegrad, intN, intValue, intY, intX)] += -tenOutgrad[intOffset] / SIZE_1(tenOne);
                                    }
                                }
                            } else {
                                for (int intValue = 0; intValue < SIZE_1(tenOne); intValue += 1) {
                                    if (fltOne[intValue] >= 0.0f) {
                                        tenOnegrad[OFFSET_4(tenOnegrad, intN, intValue, intY, intX)] += +tenOutgrad[intOffset] / SIZE_1(tenOne);
                                    } else {
                                        tenOnegrad[OFFSET_4(tenOnegrad, intN, intValue, intY, intX)] += -tenOutgrad[intOffset] / SIZE_1(tenOne);
                                    }
                                }
                            }

                            intOffset += SIZE_2(tenOutgrad) * SIZE_3(tenOutgrad);
                        }
                    }
                } }
            ''', {
                'intChans': tenOne.shape[1],
                'tenOne': tenOne,
                'tenTwo': tenTwo,
                'tenOutgrad': tenOutgrad,
                'tenOnegrad': tenOnegrad,
                'tenTwograd': tenTwograd
            }))(
                grid=tuple([int(((tenOnegrad.shape[0] * tenOnegrad.shape[2] * tenOnegrad.shape[3]) + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[cuda_int32(tenOnegrad.shape[0] * tenOnegrad.shape[2] * tenOnegrad.shape[3]), tenOne.data_ptr(), tenTwo.data_ptr(), tenOutgrad.data_ptr(), tenOnegrad.data_ptr(), tenTwograd.data_ptr()],
                stream=collections.namedtuple('Stream', 'ptr')(torch.cuda.current_stream().cuda_stream)
            )
        # end

        if tenTwograd is not None:
            cuda_launch(cuda_kernel('costvol_twograd', '''
                extern "C" __global__ void __launch_bounds__(512) costvol_twograd(
                    const int n,
                    const {{type}}* __restrict__ tenOne,
                    const {{type}}* __restrict__ tenTwo,
                    const {{type}}* __restrict__ tenOutgrad,
                    {{type}}* __restrict__ tenOnegrad,
                    {{type}}* __restrict__ tenTwograd
                ) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
                    const int intN = ( intIndex / SIZE_3(tenTwograd) / SIZE_2(tenTwograd) ) % SIZE_0(tenTwograd);
                    const int intC = -1;
                    const int intY = ( intIndex / SIZE_3(tenTwograd)                      ) % SIZE_2(tenTwograd);
                    const int intX = ( intIndex                                           ) % SIZE_3(tenTwograd);

                    {{type}} fltOne[{{intChans}}];

                    for (int intValue = 0; intValue < SIZE_1(tenOne); intValue += 1) {
                        fltOne[intValue] = VALUE_4(tenOne, intN, intValue, intY, intX);
                    }

                    int intOffset = OFFSET_4(tenOutgrad, intN, 0, intY, intX);

                    for (int intOy = intY - 4; intOy <= intY + 4; intOy += 1) {
                        for (int intOx = intX - 4; intOx <= intX + 4; intOx += 1) {
                            if ((intOy >= 0) && (intOy < SIZE_2(tenOutgrad)) && (intOx >= 0) && (intOx < SIZE_3(tenOutgrad))) {
                                for (int intValue = 0; intValue < SIZE_1(tenOne); intValue += 1) {
                                    if (fltOne[intValue] - VALUE_4(tenTwo, intN, intValue, intOy, intOx) >= 0.0f) {
                                        atomicAdd(&tenTwograd[OFFSET_4(tenTwograd, intN, intValue, intOy, intOx)], -tenOutgrad[intOffset] / SIZE_1(tenOne));
                                    } else {
                                        atomicAdd(&tenTwograd[OFFSET_4(tenTwograd, intN, intValue, intOy, intOx)], +tenOutgrad[intOffset] / SIZE_1(tenOne));
                                    }
                                }
                            } else {
                                // ...
                            }

                            intOffset += SIZE_2(tenOutgrad) * SIZE_3(tenOutgrad);
                        }
                    }
                } }
            ''', {
                'intChans': tenOne.shape[1],
                'tenOne': tenOne,
                'tenTwo': tenTwo,
                'tenOutgrad': tenOutgrad,
                'tenOnegrad': tenOnegrad,
                'tenTwograd': tenTwograd
            }))(
                grid=tuple([int(((tenTwograd.shape[0] * tenTwograd.shape[2] * tenTwograd.shape[3]) + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[cuda_int32(tenTwograd.shape[0] * tenTwograd.shape[2] * tenTwograd.shape[3]), tenOne.data_ptr(), tenTwo.data_ptr(), tenOutgrad.data_ptr(), tenOnegrad.data_ptr(), tenTwograd.data_ptr()],
                stream=collections.namedtuple('Stream', 'ptr')(torch.cuda.current_stream().cuda_stream)
            )
        # end

        return tenOnegrad, tenTwograd, None, None
    # end
# end
