#
# Vadd
#
# Element wise addition of three vectors at a time (R=A+B+C)
# Asks the user to select a device at runtime
#
# History: Initial version based on vadd.c, written by Tim Mattson, June 2011
#          Ported to C++ Wrapper API by Benedict Gaster, September 2011
#          Updated to C++ Wrapper API v1.2 by Tom Deakin and Simon McIntosh-Smith, October 2012
#          Ported to Python by Tom Deakin, July 2013
#          Ported to Julia by Jake Bolewski, Nov 2013

using OpenCL

# tolerance used in floating point comparisons
TOL = 1e-3

# length of vectors a, b, c
LENGTH = 1024

# Kernel: vadd
#
# To compute the elementwise sum r = a + b + c
#
# Input: a, b and c float vectors of length count
# Output r float vector of length count holding the sum a + b + cs

kernelsource = "
__kernel void vadd(
    __global float* a,
    __global float* b,
    __global float* c,
    __global float* r,
    const unsigned int count)
{
    unsigned int i = get_global_id(0);
    if (i < count)
        r[i] = a[i] + b[i] + c[i];
}"

# create the compute program and build it
program = cl.Program(source=kernelsource) |> cl.build!

# create a, b and c vectors and fill with random float values
# (the result array will be created when reading back from the device)
h_a = rand(Float32, LENGTH)
h_b = rand(Float32, LENGTH)
h_c = rand(Float32, LENGTH)

d_a = CLArray(h_a)
d_b = CLArray(h_b)
d_c = CLArray(h_c)

# create the output (r) buffer in device memory
d_r = CLArray{Float32}(undef, LENGTH)

# create the kernel
vadd = cl.Kernel(program, "vadd")

# execute the kernel over the entire range of the input
clcall(vadd, Tuple{CLPtr{Float32}, CLPtr{Float32}, CLPtr{Float32}, CLPtr{Float32}, Cuint},
       d_a, d_b, d_c, d_r, UInt32(LENGTH); global_size=size(h_a))

# read the results back from the compute device
h_r = Array(d_r)

# test the results
correct = 0
for i in 1:LENGTH
    tmp = h_a[i] + h_b[i] + h_c[i]
    # compute the deviation of expected and output result
    tmp -= h_r[i]
    if tmp^2 < TOL^2
        global correct += 1
    else
        println("tmp $tmp h_a $(h_a[i]) h_b $(h_b[i]) h_c $(h_c[i]) h_r $(h_r[i])")
    end
end

# summarize results
println("3 vector adds to find F=A+B+C: $correct out of $LENGTH results were correct")
