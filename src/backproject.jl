# backproject.jl

export backproject, backproject!

"""
    backproject!(image, view, plan, workarray, viewidx)
Backproject a single view.
"""
function backproject!(
    image::AbstractArray{<:RealU, 3},
    view::AbstractMatrix{<:RealU},
    plan::SPECTplan,
    workarray::Vector{Workarray},
    viewidx::Int
)

    Threads.@threads for z = 1:plan.imgsize[3] # 1:nz
        thid = Threads.threadid() # thread id
        if plan.interpidx == 1
            imrotate3!((@view plan.mumapr[:, :, z]),
                        workarray[thid].workmat_rot_1,
                        workarray[thid].workmat_rot_2,
                        (@view plan.mumap[:, :, z]),
                        plan.viewangle[viewidx],
                        workarray[thid].interp_x,
                        workarray[thid].interp_y,
                        workarray[thid].workvec_rot_x,
                        workarray[thid].workvec_rot_y)

        else
            imrotate3!((@view plan.mumapr[:, :, z]),
                        workarray[thid].workmat_rot_1,
                        workarray[thid].workmat_rot_2,
                        (@view plan.mumap[:, :, z]),
                        plan.viewangle[viewidx])

        end

    end

    # adjoint of convolving img with psf and applying attenuation map
    Threads.@threads for y = 1:plan.imgsize[2] # 1:ny
        thid = Threads.threadid() # thread id
        # account for half of the final slice thickness
        scale3dj!(workarray[thid].exp_mumapr, plan.mumapr, y, -0.5)
        for j = 1:y
            plus3dj!(workarray[thid].exp_mumapr, plan.mumapr, j)
        end

        broadcast!(*, workarray[thid].exp_mumapr, workarray[thid].exp_mumapr, - plan.dy)

        broadcast!(exp, workarray[thid].exp_mumapr, workarray[thid].exp_mumapr)

        fft_conv_adj!((@view plan.imgr[:, y, :]),
                       workarray[thid].workmat_fft,
                       workarray[thid].workvec_fft_1,
                       workarray[thid].workvec_fft_2,
                       view,
                       (@view plan.psfs[:, :, y, viewidx]),
                       plan.pad_fft,
                       workarray[thid].img_compl,
                       workarray[thid].ker_compl,
                       workarray[thid].fft_plan,
                       workarray[thid].ifft_plan)

        mul3dj!(plan.imgr, workarray[thid].exp_mumapr, y)
    end

    # adjoint of rotating image
    Threads.@threads for z = 1:plan.imgsize[3] # 1:nz
        thid = Threads.threadid()
        if plan.interpidx == 1
            imrotate3_adj!((@view image[:, :, z]),
                           workarray[thid].workmat_rot_1,
                           workarray[thid].workmat_rot_2,
                           (@view plan.imgr[:, :, z]),
                           plan.viewangle[viewidx],
                           workarray[thid].interp_x,
                           workarray[thid].interp_y,
                           workarray[thid].workvec_rot_x,
                           workarray[thid].workvec_rot_y)

        else
            imrotate3_adj!((@view image[:, :, z]),
                           workarray[thid].workmat_rot_1,
                           workarray[thid].workmat_rot_2,
                           (@view plan.imgr[:, :, z]),
                           plan.viewangle[viewidx])
        end
    end
    return image
end



"""
    backproject!(image, views, plan, workarray; index)
Backproject multiple views into `image`.
Array `image` is not initialized to zero; caller must do that.
"""
function backproject!(
    image::AbstractArray{<:RealU, 3},
    views::AbstractArray{<:RealU, 3},
    plan::SPECTplan,
    workarray::Vector{Workarray};
    index::AbstractVector{<:Int} = 1:plan.nview, # all views
)

    # loop over each view index
    for i in index
        backproject!(plan.add_img, (@view views[:, :, i]), plan, workarray, i)
        broadcast!(+, image, image, plan.add_img)
    end
end



"""
    image = backproject(plan, workarray, views ; kwargs...)
SPECT backproject `views`; this allocates the returned 3D array.
"""
function backproject(
    plan::SPECTplan,
    workarray::Vector{Workarray},
    views::AbstractArray{<:RealU, 3} ;
    kwargs...,
)
    image = zeros(plan.T, plan.imgsize)
    backproject!(image, views, plan, workarray; kwargs...)
    return image
end


"""
    image = backproject(views, mumap, psfs, dy; interpidx, kwargs...)
SPECT backproject `views` using attenuation map `mumap` and PSF array `psfs` for pixel size `dy`.
This method initializes the `plan` and `workarray` as a convenience.
Most users should use `backproject!` instead after initializing those, for better efficiency.
"""
function backproject(
    views::AbstractArray{<:RealU, 3},
    mumap::AbstractArray{<:RealU, 3}, # [nx,ny,nz] attenuation map, must be 3D, possibly zeros()
    psfs::AbstractArray{<:RealU, 4},
    dy::RealU;
    interpidx::Int = 2,
    kwargs...,
)

    plan = SPECTplan(mumap, psfs, dy; interpidx, kwargs...)
    workarray = Vector{Workarray}(undef, plan.ncore)
    for i = 1:plan.ncore
        workarray[i] = Workarray(plan.T, plan.imgsize, plan.pad_fft, plan.pad_rot) # allocate
    end
    return backproject(plan, workarray, views; kwargs...)
end
