# circuit.jl: Implements quantum circuit generation for the Chimera Cognitive Architecture using Yao.jl (0.8.0).
# Purpose: Translates symbolic problem representations from planner.lisp (received via JSON) into quantum circuits
# (e.g., Hadamard and CNOT gates for optimization tasks). Includes robust error handling for invalid circuit specifications.
# Exports functions for use in main.jl. Designed for Julia 1.10.0.

using Pkg
Pkg.activate(@__DIR__)
using Yao
using Yao.ConstGate  # For standard quantum gates (H, X, CNOT, etc.)
using JSON

# Custom exception for quantum circuit errors
struct QuantumCircuitError <: Exception
    message::String
end

# Configuration parameters
const ALLOWED_GATES = Set([:H, :X, :CNOT, :Z, :Y, :T, :S])  # Supported quantum gates
const MAX_QUBITS = 50  # Maximum number of qubits to prevent excessive resource usage
const MAX_GATES = 1000  # Maximum number of gates to prevent overly complex circuits

# Function: validate_gate_spec
# Validates a single gate specification from JSON instructions
function validate_gate_spec(gate::Dict)
    """
    Validates a quantum gate specification from JSON instructions.
    Input: gate - Dictionary with type, qubits, and optional parameters (e.g., {"type": "H", "qubits": [1]}).
    Output: Tuple (gate_type, qubits, params) if valid, throws QuantumCircuitError if invalid.
    """
    try
        if !haskey(gate, "type") || !haskey(gate, "qubits")
            throw(QuantumCircuitError("Gate specification missing 'type' or 'qubits' field"))
        end
        gate_type = Symbol(gate["type"])
        if !(gate_type in ALLOWED_GATES)
            throw(QuantumCircuitError("Unsupported gate type: $gate_type"))
        end
        qubits = gate["qubits"]
        if !isa(qubits, Vector) || isempty(qubits) || any(q -> !isa(q, Integer) || q <= 0, qubits)
            throw(QuantumCircuitError("Invalid qubits: must be a non-empty vector of positive integers"))
        end
        if maximum(qubits) > MAX_QUBITS
            throw(QuantumCircuitError("Qubit index exceeds maximum: $MAX_QUBITS"))
        end
        # Validate gate-specific requirements
        if gate_type == :CNOT && length(qubits) != 2
            throw(QuantumCircuitError("CNOT gate requires exactly two qubits"))
        elseif gate_type in [:H, :X, :Z, :Y, :T, :S] && length(qubits) != 1
            throw(QuantumCircuitError("Single-qubit gate $gate_type requires exactly one qubit"))
        end
        params = get(gate, "params", Dict())  # Optional parameters (e.g., for parameterized gates)
        return (gate_type, qubits, params)
    catch e
        if e isa QuantumCircuitError
            @error "Gate validation error: $(e.message)"
            throw(e)
        else
            @error "Unexpected error validating gate: $e"
            throw(QuantumCircuitError("Failed to validate gate specification: $e"))
        end
    end
end

# Function: create_gate
# Creates a Yao.jl gate from a validated specification
function create_gate(spec::Tuple)
    """
    Creates a Yao.jl gate from a validated specification.
    Input: spec - Tuple (gate_type, qubits, params) from validate_gate_spec.
    Output: Yao gate block or throws QuantumCircuitError.
    """
    try
        (gate_type, qubits, params) = spec
        if gate_type == :H
            return put(qubits[1] => H)
        elseif gate_type == :X
            return put(qubits[1] => X)
        elseif gate_type == :CNOT
            return control(qubits[1], qubits[2] => X)
        elseif gate_type == :Z
            return put(qubits[1] => Z)
        elseif gate_type == :Y
            return put(qubits[1] => Y)
        elseif gate_type == :T
            return put(qubits[1] => T)
        elseif gate_type == :S
            return put(qubits[1] => S)
        else
            throw(QuantumCircuitError("Unsupported gate type in creation: $gate_type"))
        end
    catch e
        if e isa QuantumCircuitError
            @error "Gate creation error: $(e.message)"
            throw(e)
        else
            @error "Unexpected error creating gate: $e"
            throw(QuantumCircuitError("Failed to create gate: $e"))
        end
    end
end

# Function: create_circuit
# Creates a quantum circuit from a list of gate specifications
function create_circuit(gate_specs::Vector, n_qubits::Int)
    """
    Creates a Yao.jl quantum circuit from a list of gate specifications.
    Input: gate_specs - Vector of gate specification dictionaries.
           n_qubits - Number of qubits in the circuit.
    Output: Yao circuit block or throws QuantumCircuitError.
    """
    try
        if n_qubits <= 0 || n_qubits > MAX_QUBITS
            throw(QuantumCircuitError("Invalid number of qubits: must be between 1 and $MAX_QUBITS"))
        end
        if isempty(gate_specs)
            throw(QuantumCircuitError("No gates specified for circuit"))
        end
        if length(gate_specs) > MAX_GATES
            throw(QuantumCircuitError("Number of gates exceeds maximum: $MAX_GATES"))
        end
        gates = [create_gate(validate_gate_spec(spec)) for spec in gate_specs]
        circuit = chain(n_qubits, gates...)
        @info "Created quantum circuit with $n_qubits qubits and $(length(gates)) gates"
        return circuit
    catch e
        if e isa QuantumCircuitError
            @error "Circuit creation error: $(e.message)"
            throw(e)
        else
            @error "Unexpected error creating circuit: $e"
            throw(QuantumCircuitError("Failed to create circuit: $e"))
        end
    end
end

# Function: execute_circuit
# Executes a quantum circuit and returns measurement results
function execute_circuit(circuit, n_shots::Int=100)
    """
    Executes a Yao.jl quantum circuit and returns measurement results.
    Input: circuit - Yao circuit block.
           n_shots - Number of measurement shots (default: 100).
    Output: Dictionary with measurement outcomes or throws QuantumCircuitError.
    """
    try
        if n_shots <= 0
            throw(QuantumCircuitError("Number of shots must be positive"))
        end
        # Simulate circuit execution
        reg = zero_state(nqubits(circuit))
        reg |> circuit
        results = measure(reg, nshots=n_shots)
        # Convert results to a frequency dictionary
        freqs = Dict{String, Int}()
        for result in results
            bitstr = string(result, base=2, pad=nqubits(circuit))
            freqs[bitstr] = get(freqs, bitstr, 0) + 1
        end
        return Dict(
            "status" => "success",
            "results" => freqs,
            "n_shots" => n_shots
        )
    catch e
        @error "Circuit execution error: $e"
        return Dict("status" => "error", "message" => "Circuit execution failed: $e")
    end
end

# Function: generate_circuit
# Main entry point to generate and execute a quantum circuit from instructions
function generate_circuit(instruction::Dict)
    """
    Generates and executes a quantum circuit based on instructions from planner.lisp via main.jl.
    Input: instruction - Dictionary with n_qubits, gates, and optional n_shots.
    Output: JSON string with circuit execution results or error.
    """
    try
        if !haskey(instruction, "n_qubits") || !haskey(instruction, "gates")
            throw(QuantumCircuitError("Instruction missing 'n_qubits' or 'gates' field"))
        end
        n_qubits = instruction["n_qubits"]
        if !isa(n_qubits, Integer)
            throw(QuantumCircuitError("n_qubits must be an integer"))
        end
        gates = instruction["gates"]
        if !isa(gates, Vector)
            throw(QuantumCircuitError("gates must be a vector of gate specifications"))
        end
        circuit = create_circuit(gates, n_qubits)
        n_shots = get(instruction, "n_shots", 100)
        result = execute_circuit(circuit, n_shots)
        return JSON.json(result)
    catch e
        if e isa QuantumCircuitError
            @error "Circuit generation error: $(e.message)"
            return JSON.json(Dict("status" => "error", "message" => e.message))
        else
            @error "Unexpected error generating circuit: $e"
            return JSON.json(Dict("status" => "error", "message" => "Circuit generation failed: $e"))
        end
    end
end

export generate_circuit
