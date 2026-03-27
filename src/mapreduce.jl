## COV_EXCL_START

# TODO
# - block-stride loop to delay need for second kernel launch
#=
function nextwarp(dev, threads::Integer)
    ws = get_sub_group_size()
    return threads + (ws - threads % ws) % ws
end

function prevwarp(dev, threads::Integer)
    ws = get_sub_group_size()
    return threads - Base.rem(threads, ws)
end
# Reduce a value across a warp
@inline function reduce_warp(op, val)
    assume(warpsize() == 32)
    offset = Int32(1)
    while offset < warpsize()
        val = op(val, intel_shfl_down(val, zero(val), offset))
        offset <<= 1
    end

    return val
end
=#
#=
# Reduce a value across a block, using shared memory for communication
@inline function reduce_block(op, val::T, neutral, shuffle::Val{true}) where T
    # shared mem for partial sums
    # assume(warpsize() == 32)
    shared = CLLocalArray(T, 32)

    wid, lane = fldmod1(get_local_id(), get_sub_group_size())

    # each warp performs partial reduction
    val = reduce_warp(op, val)

    # write reduced value to shared memory
    if lane == 1
        @inbounds shared[wid] = val
    end

    # wait for all partial reductions
    work_group_barrier(LOCAL_MEM_FENCE)

    # read from shared memory only if that warp existed
    val = if threadIdx().x <= fld1(get_local_size().x, warpsize())
         @inbounds shared[lane]
    else
        neutral
    end

    # final reduce within first warp
    if wid == 1
        val = reduce_warp(op, val)
    end

    return val
end
=#

@inline function reduce_block(op, val::T, neutral, shuffle::Val{false}, ::Val{maxitems}) where {T, maxitems}
    items = get_local_size()
    item = get_local_id()

    # shared mem for a complete reduction
    shared = CLLocalArray(T, (maxitems,))
    @inbounds shared[item] = val

    # perform a reduction
    d = 1
    while d < items
        work_group_barrier(LOCAL_MEM_FENCE)
        index = 2 * d * (item-1) + 1
        @inbounds if index <= items
            other_val = if index + d <= items
                shared[index+d]
            else
                neutral
            end
            shared[index] = op(shared[index], other_val)
        end
        d *= 2
    end

    # load the final value on the first thread
    if item == 1
        val = @inbounds shared[item]
    end

    return val
end

Base.@propagate_inbounds _map_getindex(args::Tuple, I) = ((args[1][I]), _map_getindex(Base.tail(args), I)...)
Base.@propagate_inbounds _map_getindex(args::Tuple{Any}, I) = ((args[1][I]),)
Base.@propagate_inbounds _map_getindex(args::Tuple{}, I) = ()

# Reduce an array across the grid. All elements to be processed can be addressed by the
# product of the two iterators `Rreduce` and `Rother`, where the latter iterator will have
# singleton entries for the dimensions that should be reduced (and vice versa).
function partial_mapreduce_grid(f, op, neutral, maxitems, Rreduce, Rother, shuffle, R::AbstractArray{T}, As...) where T
    assume(length(Rother) > 0)

    # decompose the 1D hardware indices into separate ones for reduction (across threads
    # and possibly blocks if it doesn't fit) and other elements (remaining blocks)
    threadIdx_reduce = get_local_id()
    blockDim_reduce = get_local_size()
    blockIdx_reduce, blockIdx_other = fldmod1(get_group_id(), length(Rother))
    gridDim_reduce = get_num_groups() ÷ length(Rother)

    # block-based indexing into the values outside of the reduction dimension
    # (that means we can safely synchronize threads within this block)
    iother = blockIdx_other
    @inbounds if iother <= length(Rother)
        Iother = Rother[iother]

        # load the neutral value
        Iout = CartesianIndex(Tuple(Iother)..., blockIdx_reduce)
        neutral = if neutral === nothing
            R[Iout]
        else
            neutral
        end

        val::T = op(neutral, neutral)

        # reduce serially across chunks of input vector that don't fit in a block
        ireduce = threadIdx_reduce + (blockIdx_reduce - 1) * blockDim_reduce
        while ireduce <= length(Rreduce)
            Ireduce = Rreduce[ireduce]
            J = max(Iother, Ireduce)
            val = op(val, f(_map_getindex(As, J)...))
            ireduce += blockDim_reduce * gridDim_reduce
        end

        val = reduce_block(op, val, neutral, shuffle, maxitems)

        # write back to memory
        if threadIdx_reduce == 1
            R[Iout] = val
        end
    end

    return
end

function serial_mapreduce_kernel(f, op, neutral, Rreduce, Rother, R, As)
    grid_idx = get_local_id() + (get_group_id() - 1) * get_local_size()
    @inbounds if grid_idx <= length(Rother)
        Iother = Rother[grid_idx]

        # load the neutral value
        neutral = if neutral === nothing
            R[Iother]
        else
            neutral
        end

        val = op(neutral, neutral)

        Ibegin = Rreduce[1]
        for Ireduce in Rreduce
            val = op(val, f(As[Iother + Ireduce - Ibegin]))
        end
        R[Iother] = val
    end
    return
end

## COV_EXCL_STOP

# factored out for use in tests
function serial_mapreduce_threshold(dev)
    dev.max_work_group_size * dev.max_compute_units
end

function GPUArrays.mapreducedim!(f::F, op::OP, R::WrappedCLArray{T},
                                 A::Union{AbstractArray,Broadcast.Broadcasted};
                                 init=nothing) where {F, OP, T}
    if !isa(A, Broadcast.Broadcasted)
        # XXX: Base.axes isn't defined anymore for Broadcasted, breaking this check
        Base.check_reducedims(R, A)
    end
    length(A) == 0 && return R # isempty(::Broadcasted) iterates
    dev = cl.device()

    # be conservative about using shuffle instructions
    #=
    shuffle = T <: Union{Bool,
                         UInt8, UInt16, UInt32, UInt64, UInt128,
                         Int8, Int16, Int32, Int64, Int128,
                         Float16, Float32, Float64,
                         ComplexF16, ComplexF32, ComplexF64}
    =#
    shuffle = false
    # add singleton dimensions to the output container, if needed
    if ndims(R) < ndims(A)
        dims = Base.fill_to_length(size(R), 1, Val(ndims(A)))
        R = reshape(R, dims)
    end

    # iteration domain, split in two: one part covers the dimensions that should
    # be reduced, and the other covers the rest. combining both covers all values.
    Rall = CartesianIndices(axes(A))
    Rother = CartesianIndices(axes(R))
    Rreduce = CartesianIndices(ifelse.(axes(A) .== axes(R), Ref(Base.OneTo(1)), axes(A)))
    # NOTE: we hard-code `OneTo` (`first.(axes(A))` would work too) or we get a
    #       CartesianIndices object with UnitRanges that behave badly on the GPU.
    @assert length(Rall) == length(Rother) * length(Rreduce)
    @assert length(Rother) > 0

    # If `Rother` is large enough, then a naive loop is more efficient than partial reductions.
    # @info serial_mapreduce_threshold(dev)
    
    if !contains(string(cl.device()), "zink") && length(Rother) >= serial_mapreduce_threshold(dev) || contains(string(cl.platform()), "Portable Computing Language")
        args = (f, op, init, Rreduce, Rother, R, A)
        kernel = @opencl launch=false serial_mapreduce_kernel(args...)
        wg_info = cl.work_group_info(kernel.fun, dev)
        local_size = wg_info.size
        global_size = cld(length(Rother), local_size) * local_size
        # @info local_size global_size
        kernel(args...; local_size, global_size)
        return R
    end
    
    # how many threads do we want?
    #
    # threads in a block work together to reduce values across the reduction dimensions;
    # we want as many as possible to improve algorithm efficiency and execution occupancy.

    wanted_threads = shuffle ? nextwarp(dev, length(Rreduce)) : length(Rreduce)
    function compute_threads(max_threads)
        if wanted_threads > max_threads
            shuffle ? prevwarp(dev, max_threads) : max_threads
        else
            wanted_threads
        end
    end
    max_lmem_elements = dev.local_mem_size ÷ sizeof(T)
    max_items = min(dev.max_work_group_size,
                    compute_threads(max_lmem_elements ÷ 2))
    
    # how many threads can we launch?
    #
    # we might not be able to launch all those threads to reduce each slice in one go.
    # that's why each threads also loops across their inputs, processing multiple values
    # so that we can span the entire reduction dimension using a single thread block.
    # @info f op init Rreduce Rother Val(shuffle) R A shuffle
    kernel = @opencl launch=false partial_mapreduce_grid(f, op, init, Val(max_items), Rreduce, Rother, Val(shuffle), R, A)
    # compute_shmem(threads) = shuffle ? 0 : threads*sizeof(T)
    # kernel_config = launch_configuration(kernel.fun; shmem=compute_shmem∘compute_threads)
    wg_info = cl.work_group_info(kernel.fun, dev)
    reduce_threads = compute_threads(wg_info.size)
    # reduce_shmem = compute_shmem(reduce_threads)

    # how many blocks should we launch?
    #
    # even though we can always reduce each slice in a single thread block, that may not be
    # optimal as it might not saturate the GPU. we already launch some blocks to process
    # independent dimensions in parallel; pad that number to ensure full occupancy.
    other_blocks = length(Rother)
    blocks = 1
    reduce_blocks = if other_blocks >= blocks
        1
    else
        min(cld(length(Rreduce), reduce_threads),       # how many we need at most
            cld(kernel_config.blocks, other_blocks))    # maximize occupancy
    end

    # determine the launch configuration
    local_size = reduce_threads
    # shmem = reduce_shmem
    global_size = reduce_blocks*other_blocks*reduce_threads
    # perform the actual reduction
    if reduce_blocks == 1
        # we can cover the dimensions to reduce using a single block
        # @info local_size global_size
        kernel(f, op, init, Val(local_size), Rreduce, Rother, Val(shuffle), R, A; local_size, global_size)
    else
        @warn "not here"
        #=
        # TODO: provide a version that atomically reduces from different blocks

        # temporary empty array whose type will match the final partial array
	    partial = similar(R, ntuple(_ -> 0, Val(ndims(R)+1)))

        # NOTE: we can't use the previously-compiled kernel, or its launch configuration,
        #       since the type of `partial` might not match the original output container
        #       (e.g. if that was a view).
        partial_kernel = @cuda launch=false partial_mapreduce_grid(f, op, init, Rreduce, Rother, Val(shuffle), partial, A)
        partial_kernel_config = launch_configuration(partial_kernel.fun; shmem=compute_shmem∘compute_threads)
        partial_reduce_threads = compute_threads(partial_kernel_config.threads)
        partial_reduce_shmem = compute_shmem(partial_reduce_threads)
        partial_reduce_blocks = if other_blocks >= partial_kernel_config.blocks
            1
        else
            min(cld(length(Rreduce), partial_reduce_threads),
                cld(partial_kernel_config.blocks, other_blocks))
        end
        partial_threads = partial_reduce_threads
        partial_shmem = partial_reduce_shmem
        partial_blocks = partial_reduce_blocks*other_blocks

        partial = similar(R, (size(R)..., partial_reduce_blocks))
        if init === nothing
            # without an explicit initializer we need to copy from the output container
            partial .= R
        end

        partial_kernel(f, op, init, Rreduce, Rother, Val(shuffle), partial, A;
                       threads=partial_threads, blocks=partial_blocks, shmem=partial_shmem)

        GPUArrays.mapreducedim!(identity, op, R, partial; init)
        =#
    end

    return R
end
