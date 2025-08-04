# Dockerfile for containerizing the Chimera Cognitive Architecture
# Purpose: Sets up a container with Julia 1.10.0, SBCL, Quicklisp, and CUDA 12.2 to run the
# symbolic reasoning (Lisp), parallel computation (Julia with Flux.jl and CUDA.jl), and
# quantum modeling (Yao.jl) components. Copies the repository, installs dependencies,
# configures environment variables from .env, and runs planner.lisp and main.jl.
# Exposes port 5000 for Lisp-Julia socket communication. Includes error handling for
# missing dependencies.

# Use NVIDIA CUDA base image with Ubuntu 20.04 for CUDA 12.2
FROM nvidia/cuda:12.2.0-devel-ubuntu20.04

# Set working directory
WORKDIR /app

# Install system dependencies and tools
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    sbcl \
    && rm -rf /var/lib/apt/lists/*

# Install Julia 1.10.0
RUN curl -L https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-1.10.0-linux-x86_64.tar.gz -o julia.tar.gz \
    && tar -xvzf julia.tar.gz \
    && mv julia-1.10.0 /opt/julia \
    && ln -s /opt/julia/bin/julia /usr/local/bin/julia \
    && rm julia.tar.gz

# Install Quicklisp
RUN curl -O https://beta.quicklisp.org/quicklisp.lisp \
    && sbcl --non-interactive --load quicklisp.lisp \
        --eval '(quicklisp-quickstart:install)' \
        --eval '(ql:add-to-init-file)' \
        --eval '(quit)' \
    && rm quicklisp.lisp

# Copy the repository
COPY . /app

# Install Julia dependencies with specific versions
RUN julia -e 'using Pkg; \
    Pkg.activate("."); \
    Pkg.add([ \
        Pkg.PackageSpec(name="Flux", version="0.14.0"), \
        Pkg.PackageSpec(name="Yao", version="0.8.0"), \
        Pkg.PackageSpec(name="CUDA", version="5.4.0"), \
        Pkg.PackageSpec(name="Test"), \
        Pkg.PackageSpec(name="JSON") \
    ]); \
    Pkg.resolve(); \
    Pkg.instantiate()' || { echo "Julia package installation failed"; exit 1; }

# Install Quicklisp dependencies
RUN sbcl --non-interactive --load /root/quicklisp/setup.lisp \
    --eval '(ql:quickload :(:fiveam :usocket :jsown :cl-ppcre))' \
    || { echo "Quicklisp dependency installation failed"; exit 1; }

# Verify CUDA availability
RUN nvidia-smi || { echo "NVIDIA drivers not found; CUDA may not function"; exit 1; } \
    && julia -e 'using CUDA; if !CUDA.functional(); println("CUDA is not functional; falling back to CPU"); end'

# Copy and configure environment variables
COPY .env.example .env
ENV JULIA_HOST=localhost
ENV JULIA_PORT=5000
ENV QUANTUM_API_KEY=""
RUN if [ -f .env ]; then \
        export $(grep -v '^#' .env | xargs); \
    else \
        echo "Warning: .env file not found; using default environment variables"; \
    fi

# Expose port for Lisp-Julia communication
EXPOSE 5000

# Healthcheck to verify Julia server is running
HEALTHCHECK --interval=30s --timeout=3s \
    CMD curl -f http://localhost:5000 || exit 1

# Run the application
CMD julia --project=. backend/julia/main.jl & \
    sleep 5 && \
    sbcl --load backend/lisp/planner.lisp \
    || { echo "Failed to start Chimera Cognitive Architecture"; exit 1; }
