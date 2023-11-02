## Many-to-many Splatting for Efficient Video Frame Interpolation (CVPR'22)

[Ping Hu](http://cs-people.bu.edu/pinghu/), [Simon Niklaus](https://sniklaus.com/), [Stan Sclaroff](http://www.cs.bu.edu/fac/sclaroff/), [Kate Saenko](http://ai.bu.edu/ksaenko.html)


[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Hu_Many-to-Many_Splatting_for_Efficient_Video_Frame_Interpolation_CVPR_2022_paper.pdf)] [[Poster](https://cs-people.bu.edu/pinghu/M2M_files/poster.pdf)] [[Slides](https://cs-people.bu.edu/pinghu/M2M_files/slides.pdf)] [[Video](https://cs-people.bu.edu/pinghu/M2M_files/video.mp4)]


Motion-based video frame interpolation commonly relies on optical flow to warp pixels from the inputs to the desired interpolation instant. Yet due to the inherent challenges of motion estimation (e.g. occlusions and discontinuities), most state-of-the-art interpolation approaches require subsequent refinement of the warped result to generate satisfying outputs, which drastically decreases the efficiency for multi-frame interpolation. In this work, we propose a fully differentiable Many-to-Many (M2M) splatting framework to interpolate frames efficiently. Specifically, given a frame pair, we estimate multiple bidirectional flows to directly forward warp the pixels to the desired time step, and then fuse any overlapping pixels. In doing so, each source pixel renders multiple target pixels and each target pixel can be synthesized from a larger area of visual context. This establishes a many-to-many splatting scheme with robustness to artifacts like holes. Moreover, for each input frame pair, M2M only performs motion estimation once and has a minuscule computational overhead when interpolating an arbitrary number of in-between frames, hence achieving fast multi-frame interpolation. We conducted extensive experiments to analyze M2M, and found that it significantly improves efficiency while maintaining high effectiveness.

<p align="center"> 
    <a><img src="./image.png" height="264"/></a>        
</p>

****Left**: ×8 interpolation on the “2K” version of X-TEST; **Right**: Runtime for interpolating 640×480 video frames.*
<br>
**Evaluated on a single Titan X GPU.*


### Requirements:
1. Linux
2. Python 3.7
3. Pytorch 1.8.0
4. NVIDIA GPU + CUDA 10.0

### Testing M2M

see [TEST_README.md](./Test/README.md)

### Training M2M

see [TRAIN_README.md](./Train/README.md)

### Citation
If you find M2M is helpful in your research, please consider citing:

    @InProceedings{hu2022m2m,
    title={Many-to-many Splatting for Efficient Video Frame Interpolation},
    author={Hu, Ping and Niklaus, Simon and Sclaroff, Stan and Saenko, Kate},
    journal={CVPR},
    year={2022}
    }

### Disclaimer

This is a pytorch re-implementation of M2M, please refer to the original paper ([Many-to-many Splatting for Efficient Video Frame Interpolation](https://arxiv.org/pdf/2204.03513.pdf](https://openaccess.thecvf.com/content/CVPR2022/papers/Hu_Many-to-Many_Splatting_for_Efficient_Video_Frame_Interpolation_CVPR_2022_paper.pdf)) for comparisons.



