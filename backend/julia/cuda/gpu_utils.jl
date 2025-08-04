# gpu_utils.jl: Manages GPU-accelerated computations for the Chimera Cognitive Architecture using CUDA.jl (5.4.0).
# Purpose: Implements functions to move neural network computations from neural/ (network.jl, training.jl) to GPU with CPU
# fallback if GPU is unavailable. Optimizes memory usage and includes robust error handling for CUDA driver issues or
# insufficient memory. Exports functions for use in neural/. Designed for Julia 1.10.0.

using Pkg
Pkg.activate(@__DIR__)
using CUDA
using Flux
using JSON

# Custom exception for GPU-related errors
struct GPUError <: Exception
    message::String
end

# Configuration parameters
const MAX_MEMORY_USAGE = 0.8  # Maximum fraction of GPU memory to use (80%)
const MIN_MEMORY_AVAILABLE = 512 * 1024^2  # Minimum free memory required (512 MB)
const ALLOWED_ARRAY_TYPES = Set([Matrix{Float32}, Vector{Float32}])  # Supported array types for GPU

# Function: check_gpu_availability
# Checks if GPU is available and has sufficient memory
function check_gpu_availability()
    """
    Checks if a CUDA-capable GPU is available and has sufficient memory.
    Output: true if GPU is available and meets memory requirements, false otherwise.
    Throws: GPUError if CUDA driver is misconfigured.
    """
    try
        if !CUDA.functional()
            @warn "CUDA is not functional; falling back to CPU"
            return false
        end
        mem_info = CUDA.memory_status()
        free_mem = CUDA.available_memory()
        if free_mem < MIN_MEMORY_AVAILABLE
            @warn "Insufficient GPU memory: $free_mem bytes available, $MIN_MEMORY_AVAILABLE required"
            return false
        end
        total_mem = CUDA.total_memory()
        if free_mem / total_mem < (1 - MAX_MEMORY_USAGE)
            @warn "GPU memory usage exceeds threshold: $(free_mem / total_mem * 100)% free"
            return false
        end
        @info "GPU available with $free_mem bytes free"
        return true
    catch e
        @error "CUDA driver error: $e"
        throw(GPUError("CUDA driver misconfigured or unavailable: $e"))
    end
end

# Function: move_to_gpu
# Moves a neural network or data to GPU with CPU fallback
function move_to_gpu(obj)
    """
    Moves a neural network (Flux Chain) or data (array) to GPU with CPU fallback.
    Input: obj - Flux Chain or array (Matrix/Vector of Float32).
    Output: GPU-moved object or original object if CPU fallback is used.
    Throws: GPUError for invalid input types or CUDA issues.
    Optimization: Ensures type compatibility and checks memory availability before transfer.
    """
    try
        if !check_gpu_availability()
            @info "Using CPU fallback for $obj"
            return obj
        end
        if obj isa Chain
            @info "Moving Flux Chain to GPU"
            return obj |> gpu
        elseif any(isa(obj, T) for T in ALLOWED_ARRAY_TYPES)
            @info "Moving array to GPU: $(size(obj))"
            return CuArray(obj)
        else
            throw(GPUError("Unsupported type for GPU transfer: $(typeof(obj))"))
        end
    catch e
        if e isa GPUError
            @error "GPU transfer error: $(e.message)"
            throw(e)
        else
            @error "Unexpected error moving to GPU: $e"
            @info "Falling back to CPU"
            return obj
        end
    end
end

# Function: optimize_memory
# Optimizes GPU memory usage by releasing unused memory
function optimize_memory()
    """
    Optimizes GPU memory by triggering garbage collection and releasing unused CUDA memory.
    Output: true if successful, false if GPU is unavailable.
    Optimization: Uses CUDA.reclaim() and GC.gc() to free memory, minimizing fragmentation.
    """
    try
        if !check_gpu_availability()
            return false
        end
        CUDA.reclaim()
        GC.gc(true)  # Force garbage collection with CUDA-aware cleanup
        @info "GPU memory optimized: $(CUDA.available_memory()) bytes free"
        return true
    catch e
        @error "Memory optimization error: $e"
        return false
    end
end

# Function: execute_gpu_computation
# Executes a GPU-accelerated computation for a neural network
function execute_gpu_computation(network, data::Dict)
    """
    Executes a GPU-accelerated computation for a neural network with input data.
    Input: network - Flux Chain (from network.jl).
           data - Dictionary with input data (e.g., {"input": Matrix{Float32}, "target": Matrix{Float32}}).
    Output: JSON string with computation results or error.
    Optimization: Moves network and data to GPU, optimizes memory, and falls back to CPU if needed.
    """
    try
        if !haskey(data, "input") || !haskey(data, "target")
            throw(GPUError("Data missing 'input' or 'target' field"))
        end
        input = data["input"]
        target = data["target"]
        if !any(isa(input, T) for T in ALLOWED_ARRAY_TYPES) || !any(isa(target, T) for T in ALLOWED_ARRAY_TYPES)
            throw(GPUError("Invalid data type: input and target must be Matrix/Vector of Float32"))
        end

        # Optimize memory before computation
        optimize_memory()

        # Move network and data to GPU if possible
        gpu_network = move_to_gpu(network)
        gpu_input = move_to_gpu(input)
        gpu_target = move_to_gpu(target)

        # Perform forward pass
        output = gpu_network(gpu_input)
        loss = mean((output .- gpu_target) .^ 2)  # Example: MSE loss

        # Move results back to CPU for serialization
        cpu_output = output isa CuArray ? Array(output) : output
        cpu_loss = loss isa CuArray ? Array(loss)[1] : loss

        @info "GPU computation completed: loss = $cpu_loss"
        return JSON.json(Dict(
            "status" => "success",
            "output" => cpu_output,
            "loss" => cpu_loss
        ))
    catch e
        if e isa GPUError
            @error "GPU computation error: $(e.message)"
            return JSON.json(Dict("status" => "error", "message" => e.message))
        else
            @error "Unexpected GPU computation error: $e"
            return JSON.json(Dict("status" => "error", "message" => "GPU computation failed: $e"))
        end
    finally
        optimize_memory()  # Clean up after computation
    end
end

export check_gpu_availability, move_to_gpu, optimize_memory, execute_gpu_computation
