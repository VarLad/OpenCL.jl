# list

export CmdBuffer, execute!

mutable struct CmdBuffer
    handle::cl_command_buffer_khr

    function CmdBuffer(queues::Vector{CmdQueue}, properties)        

        properties = Ref{cl_command_buffer_properties_khr}()
        
        err_code = Ref{cl_uint}()
        
        obj = new(clCreateCommandBufferKHR(length(queues), pointer(queues), properties, err_code))

        finalizer(obj) do obj
            clReleaseCommandBuffer(obj.handle)
        end
        
        obj
    end
end

Base.unsafe_convert(::Type{cl_command_buffer_khr}, list::CmdBuffer) = list.handle

Base.:(==)(a::CmdBuffer, b::CmdBuffer) = a.handle == b.handle
Base.hash(e::CmdBuffer, h::UInt) = hash(e.handle, h)

Base.close(list::CmdBuffer) = clFinalizeCommandBufferKHR(list)

# Base.reset(list::CmdBuffer) = zeCommandListReset(list)

"""
    CmdBuffer(dev::Device, ...) do list
        append_...!(list)
    end

Create a command buffer, passing in a do block that appends operations.
The list is then closed and can be used immediately, e.g. for execution.

"""
function CmdBuffer(f::Base.Callable, args...; kwargs...)
    list = CmdBuffer(args...; kwargs...)
    f(list)
    close(list)
    return list
end

function execute!(queues::Vector{CmdQueue}, cmdbuffer::CmdBuffer; event_wait_list::Vector{Event}=Event[])
    event_wait_list_ptr = isempty(event_wait_list) ? C_NULL : pointer(event_wait_list)
    ev = Ref{cl_event}()
    err = clEnqueueCommandBufferKHR(length(queues), pointer(queues), cmdbuffer.handle, length(event_wait_list), event_wait_list_ptr, ev)
    return err, ev[]
end

"""
    execute!(queue::CmdQueue, ...) do list
        append_...!(list)
    end

Create a command list for the device that owns `queue`, passing in a do block that appends
operations. The list is then closed and executed on the queue.
"""
function execute!(f::Base.Callable, queue::CmdQueue, fence=nothing; kwargs...)
    list = CmdBuffer(f, queue.context, queue.device, queue.ordinal; kwargs...)
    execute!(queue, [list], fence)
end
