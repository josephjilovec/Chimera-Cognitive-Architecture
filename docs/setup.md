Chimera Cognitive Architecture Setup Guide

This document provides detailed instructions for setting up and running the Chimera Cognitive Architecture, an open-source framework integrating symbolic reasoning (Common Lisp with SBCL), parallel computation (Julia with Flux.jl and CUDA.jl), and quantum modeling (Yao.jl). It covers installing dependencies, cloning the repository, configuring environment variables, running tests, and deploying via containers. Troubleshooting steps address common issues, such as CUDA driver verification. This guide is intended for advanced technical audiences, such as quantum cloud teams and AI researchers.

Prerequisites

Operating System: Ubuntu 20.04+ (or compatible Linux distribution), macOS, or Windows (with WSL2 for CUDA support).
Hardware: CPU with ≥4 cores, ≥8 GB RAM, and an NVIDIA GPU (e.g., GTX 1060 or better) for CUDA acceleration.
Internet: Required for downloading dependencies and cloning the repository.
Tools: git, curl, make, and a terminal emulator.

Installation Steps
1. Install SBCL (Steel Bank Common Lisp)
SBCL is used for the symbolic reasoning engine.

Ubuntu/Debian:sudo apt update
sudo apt install sbcl


macOS (via Homebrew):brew install sbcl


Windows (WSL2):Install Ubuntu via WSL2, then follow Ubuntu instructions.
Verify Installation:sbcl --version
# Expected: SBCL 2.x.x or later



2. Install Quicklisp
Quicklisp manages Lisp dependencies (e.g., FiveAM, usocket, jsown).

Download Quicklisp:curl -O https://beta.quicklisp.org/quicklisp.lisp


Install Quicklisp:sbcl --load quicklisp.lisp
* (quicklisp-quickstart:install)
* (ql:add-to-init-file)
* (quit)


Verify Installation:sbcl
* (ql:quickload :fiveam)
# Expected: Successfully loads FiveAM



3. Install Julia 1.10.0
Julia powers the parallel computation and quantum modeling engines.

Download Julia 1.10.0:wget https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-1.10.0-linux-x86_64.tar.gz


Extract and Install:tar -xvzf julia-1.10.0-linux-x86_64.tar.gz
sudo mv julia-1.10.0 /opt/julia
sudo ln -s /opt/julia/bin/julia /usr/local/bin/julia


Verify Installation:julia --version
# Expected: julia version 1.10.0



4. Install CUDA 12.2 (Optional for GPU Acceleration)
CUDA enables GPU-accelerated neural computations.

Check GPU Compatibility:nvidia-smi
# Expected: Lists NVIDIA GPU and driver version

If nvidia-smi fails, install NVIDIA drivers:sudo apt install nvidia-driver-<version> nvidia-utils-<version>


Install CUDA Toolkit 12.2:wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda_12.2.0_535.86.10_linux.run
sudo sh cuda_12.2.0_535.86.10_linux.run

Follow prompts to install the toolkit (exclude driver if already installed).
Set Environment Variables:echo 'export PATH=/usr/local/cuda-12.2/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc


Verify Installation:nvcc --version
# Expected: Cuda compilation tools, release 12.2



5. Install Julia Packages
Install required Julia packages: Flux.jl (0.14.0), Yao.jl (0.8.0), CUDA.jl (5.4.0).

Start Julia:julia


Install Packages:using Pkg
Pkg.add([
    Pkg.PackageSpec(name="Flux", version="0.14.0"),
    Pkg.PackageSpec(name="Yao", version="0.8.0"),
    Pkg.PackageSpec(name="CUDA", version="5.4.0"),
    Pkg.PackageSpec(name="Test"),
    Pkg.PackageSpec(name="JSON")
])


Verify Installation:using Flux, Yao, CUDA
println(Flux.__version__)  # Expected: v0.14.0
println(Yao.__version__)   # Expected: v0.8.0
println(CUDA.__version__)  # Expected: v5.4.0



6. Clone the Repository
Clone the Chimera Cognitive Architecture repository.
git clone https://github.com/your-org/chimera-cognitive-architecture.git
cd chimera-cognitive-architecture

Note: Replace https://github.com/your-org/chimera-cognitive-architecture.git with the actual repository URL.
7. Configure Environment Variables
Copy the example environment file and configure settings.

Copy .env.example:cp .env.example .env


Edit .env:nano .env

Example .env:JULIA_HOST=localhost
JULIA_PORT=5000
QUANTUM_API_KEY=your_quantum_api_key  # Optional for quantum hardware


JULIA_HOST and JULIA_PORT: For Lisp-Julia socket communication.
QUANTUM_API_KEY: Required for quantum hardware backends (e.g., IBM Quantum, AWS Braket). Leave empty for simulator-only mode.



8. Run Tests
Run tests to verify the setup.

Lisp Tests:
sbcl --load tests/lisp/test-knowledge.lisp
sbcl --load tests/lisp/test-planner.lisp

Expected output: Test suites run with no failures, indicating successful setup of knowledge-base.lisp and planner.lisp.

Julia Tests:
julia --project=. -e 'using Pkg; Pkg.test()'

This runs test-neural.jl and test-quantum.jl, covering network.jl, training.jl, circuit.jl, and execution.jl. Expected output: All tests pass with ≥90% coverage.


9. Run the Application
Start the Julia server and Lisp planner.

Start Julia Server:
julia --project=. backend/julia/main.jl

The server listens on JULIA_HOST:JULIA_PORT (default: localhost:5000).

Run Lisp Planner:
sbcl --load backend/lisp/planner.lisp
* (execute-plan "classification")

Example output:
("{\"status\": \"success\", \"message\": \"Network created\", \"network\": \"Chain(Dense(784 => 128))\"}")



10. Containerized Setup (Optional)
For cloud deployment, use Docker and Kubernetes configurations in deploy/.

Build Docker Image:docker build -t chimera-cognitive-architecture -f deploy/Dockerfile .


Run Docker Container:docker run -p 5000:5000 --env-file .env chimera-cognitive-architecture


Kubernetes Deployment:kubectl apply -f deploy/kubernetes.yml

See deploy/aws_config.yml for AWS ECS/EC2 deployment. Refer to architecture.md#scalability-and-deployment for details.

Troubleshooting

SBCL Fails to Load:

Ensure SBCL is installed: sbcl --version.
Check Quicklisp setup: sbcl --eval '(ql:quickload :fiveam)'.
Verify backend/lisp/reasoning/ files are accessible.


Julia Package Installation Fails:

Check Julia version: julia --version.
Clear package cache: rm -rf ~/.julia.
Reinstall packages: julia -e 'using Pkg; Pkg.add("Flux")'.


CUDA Issues:

Verify NVIDIA drivers: nvidia-smi.
Check CUDA toolkit: nvcc --version.
Ensure GPU is detected: julia -e 'using CUDA; println(CUDA.functional())'.
If CUDA.functional() returns false, fallback to CPU is automatic. Update drivers or reinstall CUDA 12.2.


Socket Communication Errors:

Verify JULIA_HOST and JULIA_PORT in .env match main.jl settings.
Check if Julia server is running: netstat -tuln | grep 5000.
Ensure firewall allows TCP connections on the specified port.


Test Failures:

Run individual tests: julia tests/julia/test-neural.jl.
Check logs for specific errors (e.g., NeuralNetworkError, QuantumCircuitError).
Verify mock data in tests aligns with expected formats.


Dependency Conflicts:

For Julia: Use Pkg.status() to check versions.
For Lisp: Ensure Quicklisp loads correct versions of fiveam, usocket, jsown.



Next Steps

Explore APIs: See docs/api.md for function signatures and usage examples.
Understand Architecture: Refer to docs/architecture.md for component interactions and data flow.
Contribute: Follow the repository’s contribution guidelines and use .github/workflows/ci.yml for CI/CD integration.

This setup ensures a fully operational Chimera Cognitive Architecture, ready for development, testing, and deployment.
