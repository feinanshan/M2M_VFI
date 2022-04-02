## Many-to-many Splatting for Efficient Video Frame Interpolation (CVPR'22)


[Ping Hu](http://cs-people.bu.edu/pinghu/), [Simon Niklaus](https://sniklaus.com/welcome/), [Stan Sclaroff](http://www.cs.bu.edu/~sclaroff/), [Kate Saenko](http://ai.bu.edu/ksaenko.html/)

[[Paper]()] [[Poster]()] [[Slides]()] [[Video]()]

Motion-based video frame interpolation commonly relies on optical flow to warp pixels from the inputs to the desired interpolation instant. Yet due to the inherent challenges of motion estimation (e.g. occlusions and discontinuities), most state-of-the-art interpolation approaches require subsequent refinement of the warped result to generate satisfying outputs, which drastically decreases the efficiency for multi-frame interpolation. In this work, we propose a fully differentiable Many-to-Many (M2M) splatting framework to interpolate frames efficiently. Specifically, given a frame pair, we estimate multiple bidirectional flows to directly forward warp the pixels to the desired time step, and then fuse any overlapping pixels. In doing so, each source pixel renders multiple target pixels and each target pixel can be synthesized from a larger area of visual context. This establishes a many-to-many splatting scheme with robustness to artifacts like holes. Moreover, for each input frame pair, M2M only performs motion estimation once and has a minuscule computational overhead when interpolating an arbitrary number of in-between frames, hence achieving fast multi-frame interpolation. We conducted extensive experiments to analyze M2M, and found that it significantly improves efficiency while maintaining high effectiveness.


### Testing M2M

### Training M2M

### Citation
If you find M2M is helpful in your research, please consider citing:

    @InProceedings{hu2022m2m,
    title={Many-to-many Splatting for Efficient Video Frame Interpolation},
    author={Hu, Ping and Niklaus, Simon and Sclaroff, Stan and Saenko, Kate},
    journal={CVPR},
    year={2022}
    }

### Disclaimer


### References
