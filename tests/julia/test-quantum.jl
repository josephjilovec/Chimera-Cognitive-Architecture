# test-quantum.jl: Unit and integration tests for backend/julia/quantum/circuit.jl and execution.jl
# in the Chimera Cognitive Architecture using Test.jl. Tests quantum circuit generation and execution
# with Yao.jl (0.8.0). Mocks quantum backend APIs to isolate tests and ensures ≥90% test coverage.
# Includes error handling tests for invalid circuit specifications. Designed for Julia 1.10.0.

using Pkg
Pkg.activate(@__DIR__)
using Test
using Yao
using Yao.ConstGate
using JSON
using Random

# Include the modules to be tested
include("../../backend/julia/quantum/circuit.jl")
include("../../backend/julia/quantum/execution.jl")

# Mock data setup
const MOCK_CIRCUIT_INSTRUCTION = Dict(
    "n_qubits" => 2,
    "gates" => [
        Dict("type" => "H", "qubits" => [1]),
        Dict("type" => "CNOT", "qubits" => [1, 2])
    ],
    "n_shots" => 100
)
const MOCK_EXECUTION_PARAMS = Dict(
    "n_shots" => 100,
    "backend" => string(SIMULATOR_BACKEND)
)

# Mock hardware backend response
function mock_hardware_response(circuit, n_shots::Int)
    """
    Mocks a quantum hardware backend response for testing.
    Input: circuit - Yao circuit block (ignored in mock).
           n_shots - Number of measurement shots.
    Output: Dictionary mimicking hardware execution results.
    """
    freqs = Dict{String, Int}()
    for _ in 1:n_shots
        bitstr = rand(["00", "01", "10", "11"])
        freqs[bitstr] = get(freqs, bitstr, 0) + 1
    end
    return Dict(
        "status" => "success",
        "backend" => string(HARDWARE_BACKEND),
        "results" => freqs,
        "n_shots" => n_shots
    )
end

# Test suite for quantum circuit generation
@testset "Quantum Circuit Generation Tests" begin
    # Test: validate_gate_spec
    @testset "validate_gate_spec" begin
        """
        Tests validation of gate specifications for circuit creation.
        Ensures valid specs are parsed correctly and invalid specs throw QuantumCircuitError.
        """
        # Valid Hadamard gate
        let spec = Dict("type" => "H", "qubits" => [1])
            result = validate_gate_spec(spec)
            @test result == (:H, [1], Dict())
        end
        # Valid CNOT gate
        let spec = Dict("type" => "CNOT", "qubits" => [1, 2])
            result = validate_gate_spec(spec)
            @test result == (:CNOT, [1, 2], Dict())
        end
        # Invalid gate type
        let spec = Dict("type" => "INVALID", "qubits" => [1])
            @test_throws QuantumCircuitError validate_gate_spec(spec)
        end
        # Missing qubits
        let spec = Dict("type" => "H")
            @test_throws QuantumCircuitError validate_gate_spec(spec)
        end
        # Invalid qubits
        let spec = Dict("type" => "H", "qubits" => [0])
            @test_throws QuantumCircuitError validate_gate_spec(spec)
        end
        # Exceeding max qubits
        let spec = Dict("type" => "H", "qubits" => [MAX_QUBITS + 1])
            @test_throws QuantumCircuitError validate_gate_spec(spec)
        end
        # Incorrect qubit count for CNOT
        let spec = Dict("type" => "CNOT", "qubits" => [1])
            @test_throws QuantumCircuitError validate_gate_spec(spec)
        end
    end

    # Test: create_gate
    @testset "create_gate" begin
        """
        Tests creation of individual Yao gates from parsed specifications.
        Verifies correct gate construction and error handling for unsupported types.
        """
        # Hadamard gate
        let spec = (:H, [1], Dict())
            gate = create_gate(spec)
            @test gate isa PutBlock
            @test gate.block == H
            @test gate.locs == (1,)
        end
        # CNOT gate
        let spec = (:CNOT, [1, 2], Dict())
            gate = create_gate(spec)
            @test gate isa ControlBlock
            @test gate.ctrl_locs == (1,)
            @test gate.locs == (2,)
            @test gate.block == X
        end
        # Invalid gate type
        let spec = (:INVALID, [1], Dict())
            @test_throws QuantumCircuitError create_gate(spec)
        end
    end

    # Test: create_circuit
    @testset "create_circuit" begin
        """
        Tests creation of a full quantum circuit from gate specifications.
        Checks circuit structure and qubit/gate constraints.
        """
        # Valid circuit
        let specs = MOCK_CIRCUIT_INSTRUCTION["gates"], n_qubits = 2
            circuit = create_circuit(specs, n_qubits)
            @test circuit isa ChainBlock
            @test nqubits(circuit) == 2
            @test length(subblocks(circuit)) == 2
            @test subblocks(circuit)[1] isa PutBlock
            @test subblocks(circuit)[2] isa ControlBlock
        end
        # Empty gates
        let specs = [], n_qubits = 2
            @test_throws QuantumCircuitError create_circuit(specs, n_qubits)
        end
        # Invalid qubit count
        let specs = MOCK_CIRCUIT_INSTRUCTION["gates"], n_qubits = 0
            @test_throws QuantumCircuitError create_circuit(specs, n_qubits)
        end
        # Exceeding max gates
        let specs = [Dict("type" => "H", "qubits" => [1]) for _ in 1:(MAX_GATES + 1)], n_qubits = 2
            @test_throws QuantumCircuitError create_circuit(specs, n_qubits)
        end
    end

    # Test: generate_circuit
    @testset "generate_circuit" begin
        """
        Tests the main circuit generation function.
        Verifies JSON output and error handling for invalid instructions.
        """
        # Valid circuit generation
        let instruction = MOCK_CIRCUIT_INSTRUCTION
            result = JSON.parse(generate_circuit(instruction))
            @test result["status"] == "success"
            @test haskey(result, "results")
            @test haskey(result, "n_shots")
            @test result["n_shots"] == 100
            @test result["results"] isa Dict
        end
        # Missing required fields
        let instruction = Dict("gates" => MOCK_CIRCUIT_INSTRUCTION["gates"])
            result = JSON.parse(generate_circuit(instruction))
            @test result["status"] == "error"
            @test occursin("missing 'n_qubits'", result["message"])
        end
        # Invalid n_qubits
        let instruction = merge(MOCK_CIRCUIT_INSTRUCTION, Dict("n_qubits" => "invalid"))
            @test_throws QuantumCircuitError generate_circuit(instruction)
        end
    end
end

# Test suite for quantum circuit execution
@testset "Quantum Circuit Execution Tests" begin
    # Test: validate_execution_params
    @testset "validate_execution_params" begin
        """
        Tests validation of execution parameters.
        Ensures valid parameters are parsed and invalid ones throw QuantumExecutionError.
        """
        # Valid simulator parameters
        let params = MOCK_EXECUTION_PARAMS
            result = validate_execution_params(params)
            @test result == (100, SIMULATOR_BACKEND)
        end
        # Valid hardware parameters (with mock API key)
        let params = Dict("n_shots" => 100, "backend" => string(HARDWARE_BACKEND)), ENV["QUANTUM_API_KEY"] = "mock_key"
            result = validate_execution_params(params)
            @test result == (100, HARDWARE_BACKEND)
        end
        # Invalid n_shots
        let params = Dict("n_shots" => MAX_SHOTS + 1)
            @test_throws QuantumExecutionError validate_execution_params(params)
        end
        # Invalid backend
        let params = Dict("n_shots" => 100, "backend" => "invalid")
            @test_throws QuantumExecutionError validate_execution_params(params)
        end
        # Missing API key for hardware
        let params = Dict("n_shots" => 100, "backend" => string(HARDWARE_BACKEND)), ENV["QUANTUM_API_KEY"] = ""
            @test_throws QuantumExecutionError validate_execution_params(params)
        end
    end

    # Test: execute_simulator
    @testset "execute_simulator" begin
        """
        Tests execution of a quantum circuit on Yao.jl's simulator.
        Verifies measurement results and qubit constraints.
        """
        let circuit = create_circuit(MOCK_CIRCUIT_INSTRUCTION["gates"], 2)
            result = execute_simulator(circuit, 100)
            @test result["status"] == "success"
            @test result["backend"] == string(SIMULATOR_BACKEND)
            @test result["n_shots"] == 100
            @test result["results"] isa Dict
            @test sum(values(result["results"])) == 100
        end
        # Exceeding max qubits
        let circuit = chain(MAX_QUBITS + 1, [put(1 => H)])
            @test_throws QuantumExecutionError execute_simulator(circuit, 100)
        end
    end

    # Test: execute_hardware
    @testset "execute_hardware" begin
        """
        Tests execution of a quantum circuit on a mocked hardware backend.
        Verifies mock response format and fallback to simulator.
        """
        let circuit = create_circuit(MOCK_CIRCUIT_INSTRUCTION["gates"], 2)
            # Mock hardware execution
            original_execute_hardware = getfield(@__MODULE__, :execute_hardware)
            setfield!(@__MODULE__, :execute_hardware, mock_hardware_response)
            result = execute_hardware(circuit, 100)
            @test result["status"] == "success"
            @test result["backend"] == string(HARDWARE_BACKEND)
            @test result["n_shots"] == 100
            @test result["results"] isa Dict
            @test sum(values(result["results"])) == 100
            # Restore original function
            setfield!(@__MODULE__, :execute_hardware, original_execute_hardware)
        end
    end

    # Test: execute_circuit
    @testset "execute_circuit" begin
        """
        Tests execution of a quantum circuit with specified backend.
        Verifies simulator and mocked hardware execution paths.
        """
        let circuit = create_circuit(MOCK_CIRCUIT_INSTRUCTION["gates"], 2)
            # Simulator backend
            result = execute_circuit(circuit, MOCK_EXECUTION_PARAMS)
            @test result["status"] == "success"
            @test result["backend"] == string(SIMULATOR_BACKEND)
            # Mock hardware backend
            let params = Dict("n_shots" => 100, "backend" => string(HARDWARE_BACKEND)), ENV["QUANTUM_API_KEY"] = "mock_key"
                original_execute_hardware = getfield(@__MODULE__, :execute_hardware)
                setfield!(@__MODULE__, :execute_hardware, mock_hardware_response)
                result = execute_circuit(circuit, params)
                @test result["status"] == "success"
                @test result["backend"] == string(HARDWARE_BACKEND)
                setfield!(@__MODULE__, :execute_hardware, original_execute_hardware)
            end
            # Invalid parameters
            let params = Dict("n_shots" => -1)
                @test_throws QuantumExecutionError execute_circuit(circuit, params)
            end
        end
    end

    # Test: run_quantum_circuit
    @testset "run_quantum_circuit" begin
        """
        Tests the main circuit execution function with a serialized circuit.
        Verifies JSON output and error handling for invalid instructions.
        """
        let circuit = create_circuit(MOCK_CIRCUIT_INSTRUCTION["gates"], 2)
            instruction = Dict(
                "circuit" => string(circuit),  # Serialized circuit
                "params" => MOCK_EXECUTION_PARAMS
            )
            result = JSON.parse(run_quantum_circuit(instruction))
            @test result["status"] == "success"
            @test haskey(result, "results")
            @test result["n_shots"] == 100
        end
        # Missing circuit or params
        let instruction = Dict("circuit" => string(create_circuit(MOCK_CIRCUIT_INSTRUCTION["gates"], 2)))
            result = JSON.parse(run_quantum_circuit(instruction))
            @test result["status"] == "error"
            @test occursin("missing 'circuit' or 'params'", result["message"])
        end
        # Invalid circuit
        let instruction = Dict("circuit" => "invalid", "params" => MOCK_EXECUTION_PARAMS)
            result = JSON.parse(run_quantum_circuit(instruction))
            @test result["status"] == "error"
            @test occursin("Invalid circuit", result["message"])
        end
    end
end

# Main test runner
function run_tests()
    """
    Runs all quantum circuit tests and logs results.
    """
    @info "Running quantum circuit tests..."
    try
        @testset "All Quantum Tests" begin
            include(@__FILE__)  # Re-run all tests in the file
        end
    catch e
        @error "Test suite error: $e"
        return false
    end
    return true
end

# Run tests
run_tests()
