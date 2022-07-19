# project.jl

export project, project!

"""
    project!(view, plan, workarray, image, viewidx)
SPECT projection of `image` into a single `view` with index `viewidx`.
"""
function project!(
    view::AbstractMatrix{<:RealU},
    image::AbstractArray{<:RealU, 3},
    plan::SPECTplan,
    workarray::Vector{Workarray},
    viewidx::Int
)
    # rotate image and mumap using multiple processors

    Threads.@threads for z = 1:plan.imgsize[3] # 1:nz
        thid = Threads.threadid() # thread id
#       work = workarray[thid] # todo: how to avoid repeating?
        if plan.interpidx == 1
            # rotate image and store in plan.imgr using 1D interpolation
            imrotate3!((@view plan.imgr[:, :, z]),
                        workarray[thid].workmat_rot_1,
                        workarray[thid].workmat_rot_2,
                        (@view image[:, :, z]),
                        plan.viewangle[viewidx],
                        workarray[thid].interp_x,
                        workarray[thid].interp_y,
                        workarray[thid].workvec_rot_x,
                        workarray[thid].workvec_rot_y,
                        )

            # rotate mumap and store in plan.mumapr
            imrotate3!((@view plan.mumapr[:, :, z]),
                        workarray[thid].workmat_rot_1,
                        workarray[thid].workmat_rot_2,
                        (@view plan.mumap[:, :, z]),
                        plan.viewangle[viewidx],
                        workarray[thid].interp_x,
                        workarray[thid].interp_y,
                        workarray[thid].workvec_rot_x,
                        workarray[thid].workvec_rot_y,
                        )
        else
            # rotate image and store in plan.imgr using 2d interpolation method
            imrotate3!((@view plan.imgr[:, :, z]),
                        workarray[thid].workmat_rot_1,
                        workarray[thid].workmat_rot_2,
                        (@view image[:, :, z]),
                        plan.viewangle[viewidx])

            # rotate mumap and store in plan.mumapr
            imrotate3!((@view plan.mumapr[:, :, z]),
                        workarray[thid].workmat_rot_1,
                        workarray[thid].workmat_rot_2,
                        (@view plan.mumap[:, :, z]),
                        plan.viewangle[viewidx],
                        )
        end
    end

    Threads.@threads for y = 1:plan.imgsize[2] # 1:ny
        thid = Threads.threadid() # thread id
        # account for half of the final slice thickness
        scale3dj!(workarray[thid].exp_mumapr, plan.mumapr, y, -0.5)
        for j = 1:y
            plus3dj!(workarray[thid].exp_mumapr, plan.mumapr, j)
        end

        broadcast!(*, workarray[thid].exp_mumapr, workarray[thid].exp_mumapr, - plan.dy)

        broadcast!(exp, workarray[thid].exp_mumapr, workarray[thid].exp_mumapr)
        # apply depth-dependent attenuation
        mul3dj!(plan.imgr, workarray[thid].exp_mumapr, y)

        fft_conv!((@view plan.add_img[:, y, :]),
                  workarray[thid].workmat_fft,
                  (@view plan.imgr[:, y, :]),
                  (@view plan.psfs[:, :, y, viewidx]),
                  plan.pad_fft,
                  workarray[thid].img_compl,
                  workarray[thid].ker_compl,
                  workarray[thid].fft_plan,
                  workarray[thid].ifft_plan)
    end

    # add up to get view
    for y in 1:plan.imgsize[2]
        plus3dj!(view, plan.add_img, y)
    end

    # plan.add_img[2] # why does julia allocate (on heap!?) here?

    return view
end


"""
    project!(views, image, plan, workarray; index)
Project `image` into multiple `views` with indexes `index` (defaults to `1:nview`).
"""
function project!(
    views::AbstractArray{<:RealU,3},
    image::AbstractArray{<:RealU,3},
    plan::SPECTplan,
    workarray::Vector{Workarray};
    index::AbstractVector{<:Int} = 1:plan.nview, # all views
)

    # loop over each view index
    for i in index
        project!((@view views[:,:,i]), image, plan, workarray, i)
    end
    return views
end


"""
    views = project(image, plan, workarray ; kwargs...)
SPECT forward projector that allocates and returns views.
"""
function project(
    image::AbstractArray{<:RealU,3},
    plan::SPECTplan,
    workarray::Vector{Workarray};
    kwargs...,
)
    views = zeros(plan.T, plan.imgsize[1], plan.imgsize[3], plan.nview)
    project!(views, image, plan, workarray; kwargs...)
    return views
end


"""
    views = project(image, mumap, psfs, dy; interpidx, kwargs...)
Initialize plan and workarray
"""
function project(
    image::AbstractArray{<:RealU, 3},
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
    return project(image, plan, workarray; kwargs...)
end
