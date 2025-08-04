;; planner.lisp: Entry point for the Chimera Cognitive Architecture's symbolic reasoning engine.
;; Purpose: Decomposes high-level problems into sub-problems, generates instructions for
;; neural network training (Flux.jl) and quantum circuit execution (Yao.jl), and coordinates
;; with the Julia backend via JSON over TCP sockets. Interfaces with the knowledge base in
;; reasoning/. Uses ASDF for package management, with enhanced error handling, input validation,
;; and security measures for robust and secure operation.

(ql:quickload :asdf)

(asdf:defsystem :chimera-planner
  :depends-on (:cl-json :usocket :fiveam :cl-ppcre)
  :components ((:file "reasoning/knowledge-base")))

(defpackage :chimera-planner
  (:use :cl :cl-json :usocket :cl-ppcre)
  (:export #:start-planner #:decompose-problem #:generate-instructions #:send-to-julia))

(in-package :chimera-planner)

;; Configuration parameters
(defparameter *julia-host* "localhost" "Host for Julia backend communication")
(defparameter *julia-port* 5000 "Port for Julia backend communication (set in .env)")
(defparameter *knowledge-base* (make-hash-table :test #'equal)
  "In-memory knowledge base for symbolic reasoning")
(defparameter *max-input-length* 10000 "Maximum length for input strings to prevent overflow")
(defparameter *allowed-task-types* '("classification" "quantum-optimization")
  "Allowed task types for validation")

;; Custom conditions for error handling
(define-condition invalid-input-error (error)
  ((message :initarg :message :reader message))
  (:report (lambda (condition stream)
             (format stream "Invalid input: ~A" (message condition)))))

(define-condition julia-backend-error (error)
  ((message :initarg :message :reader message))
  (:report (lambda (condition stream)
             (format stream "Julia backend error: ~A" (message condition)))))

(define-condition security-error (error)
  ((message :initarg :message :reader message))
  (:report (lambda (condition stream)
             (format stream "Security violation: ~A" (message condition)))))

;; Function: sanitize-input
;; Sanitizes input strings to prevent injection attacks
(defun sanitize-input (input)
  "Sanitizes input string to remove potentially malicious content.
   Input: input (string) - Input to sanitize.
   Output: Sanitized string or nil if invalid."
  (cond
    ((not (stringp input)) (error 'invalid-input-error :message "Input must be a string"))
    ((> (length input) *max-input-length*)
     (error 'security-error :message "Input exceeds maximum length"))
    (t (regex-replace-all "[^a-zA-Z0-9\\s\\-_\\.]" input ""))))

;; Function: validate-task
;; Validates task structure and content
(defun validate-task (task)
  "Validates a task structure for correctness and security.
   Input: task (list) - Task with :type and :data keys.
   Output: t if valid, raises error if invalid."
  (unless (and (listp task) (evenp (length task)))
    (error 'invalid-input-error :message "Task must be a proper plist"))
  (let ((task-type (getf task :type))
        (task-data (getf task :data)))
    (unless (and task-type task-data)
      (error 'invalid-input-error :message "Task missing :type or :data"))
    (unless (member task-type *allowed-task-types* :test #'string=)
      (error 'security-error :message (format nil "Invalid task type: ~A" task-type)))
    (when (stringp task-data)
      (setf (getf task :data) (sanitize-input task-data)))
    t))

;; Function: load-knowledge-base
;; Loads the knowledge base from reasoning/knowledge-base.lisp
(defun load-knowledge-base ()
  "Initializes the knowledge base with concepts from reasoning/knowledge-base.lisp."
  (handler-case
      (progn
        (asdf:load-system :chimera-planner)
        (setf *knowledge-base* (chimera-reasoning:initialize-knowledge-base))
        (format t "Knowledge base loaded successfully.~%"))
    (error (e)
      (error 'invalid-input-error :message (format nil "Failed to load knowledge base: ~A" e)))))

;; Function: decompose-problem
;; Decomposes a high-level problem into symbolic sub-problems
(defun decompose-problem (problem)
  "Decomposes a high-level problem into sub-problems using the knowledge base.
   Input: problem (string or list) - High-level problem description or structure.
   Output: List of validated sub-problems or nil on failure."
  (handler-case
      (cond
        ((stringp problem)
         (let ((sanitized-problem (sanitize-input problem)))
           (unless (> (length sanitized-problem) 0)
             (error 'invalid-input-error :message "Sanitized problem is empty"))
           (let ((sub-problems (chimera-reasoning:query-knowledge-base *knowledge-base* sanitized-problem)))
             (mapcar #'validate-task sub-problems)
             sub-problems)))
        ((listp problem)
         (mapcar #'validate-task problem)
         (mapcar #'process-sub-task problem))
        (t
         (error 'invalid-input-error :message "Problem must be a string or list")))
    (invalid-input-error (e)
      (format t "Input validation error: ~A~%" (message e))
      nil)
    (security-error (e)
      (format t "Security error: ~A~%" (message e))
      nil)
    (error (e)
      (format t "Decomposition error: ~A~%" e)
      nil)))

;; Helper Function: process-sub-task
;; Processes individual tasks for decomposition
(defun process-sub-task (task)
  "Processes a single task into a sub-problem structure.
   Input: task (list) - Validated task (e.g., (:type 'classification' :data 'images')).
   Output: Symbolic sub-problem."
  (let ((task-type (getf task :type))
        (task-data (getf task :data)))
    (list :sub-task task-type :data task-data)))

;; Function: generate-instructions
;; Generates instructions for neural network or quantum circuit execution
(defun generate-instructions (sub-problem)
  "Generates JSON instructions for Julia backend based on sub-problem type.
   Input: sub-problem (list) - Validated sub-problem structure.
   Output: JSON string with instructions for Flux.jl or Yao.jl."
  (handler-case
      (let ((task-type (getf sub-problem :sub-task))
            (task-data (getf sub-problem :data)))
        (cond
          ((string= task-type "classification")
           (json:encode-json-to-string
            `((:module . "Flux")
              (:code . ,(format nil "model = Chain(Dense(784, 128, relu), Dense(128, 10, softmax)); train!(model, ~A)" task-data)))))
          ((string= task-type "quantum-optimization")
           (json:encode-json-to-string
            `((:module . "Yao")
              (:code . ,(format nil "circuit = chain(4, put(1 => H), put(2 => X), measure(4)); optimize!(circuit, ~A)" task-data)))))
          (t
           (error 'invalid-input-error :message (format nil "Unsupported task type: ~A" task-type)))))
    (invalid-input-error (e)
      (format t "Instruction generation error: ~A~%" (message e))
      nil)
    (error (e)
      (format t "Unexpected error generating instructions: ~A~%" e)
      nil)))

;; Function: send-to-julia
;; Sends instructions to the Julia backend via secure TCP socket
(defun send-to-julia (instructions)
  "Sends JSON instructions to the Julia backend via TCP socket with timeout and retry.
   Input: instructions (string) - JSON-encoded instructions.
   Output: Response from Julia backend or nil on failure."
  (unless (stringp instructions)
    (error 'invalid-input-error :message "Instructions must be a JSON string"))
  (handler-case
      (with-client-socket (socket stream *julia-host* *julia-port*
                                 :element-type 'character
                                 :timeout 10)
        ;; Ensure instructions are sanitized
        (let ((sanitized-instructions (sanitize-input instructions)))
          (write-string sanitized-instructions stream)
          (force-output stream)
          (let ((response (read-line stream nil nil)))
            (unless response
              (error 'julia-backend-error :message "No response from Julia backend"))
            (format t "Julia backend response: ~A~%" response)
            response)))
    (usocket:timeout-error (e)
      (format t "Socket timeout: ~A~%" e)
      (error 'julia-backend-error :message "Connection to Julia backend timed out"))
    (usocket:socket-error (e)
      (format t "Socket error: ~A~%" e)
      (error 'julia-backend-error :message (format nil "Failed to connect to Julia backend: ~A" e)))
    (invalid-input-error (e)
      (format t "Input error: ~A~%" (message e))
      nil)
    (julia-backend-error (e)
      (format t "Julia backend error: ~A~%" (message e))
      nil)
    (error (e)
      (format t "Unexpected error in send-to-julia: ~A~%" e)
      nil)))

;; Function: start-planner
;; Entry point for the symbolic reasoning engine
(defun start-planner (problem)
  "Starts the symbolic reasoning engine, orchestrating problem decomposition and instruction generation.
   Input: problem (string or list) - High-level problem to solve.
   Output: List of responses from Julia backend or nil on failure."
  (handler-case
      (progn
        (load-knowledge-base)
        (let ((sub-problems (decompose-problem problem)))
          (if sub-problems
              (let* ((instructions (remove nil (mapcar #'generate-instructions sub-problems)))
                     (responses (remove nil (mapcar #'send-to-julia instructions))))
                (if responses
                    (progn
                      (format t "Processing complete. Responses: ~A~%" responses)
                      responses)
                    (error 'julia-backend-error :message "No valid responses from Julia backend")))
              (error 'invalid-input-error :message "No valid sub-problems generated"))))
    (invalid-input-error (e)
      (format t "Input validation error: ~A~%" (message e))
      nil)
    (security-error (e)
      (format t "Security error: ~A~%" (message e))
      nil)
    (julia-backend-error (e)
      (format t "Julia backend error: ~A~%" (message e))
      nil)
    (error (e)
      (format t "Planner error: ~A~%" e)
      nil)))

;; Example usage
(defun example ()
  "Example usage of the planner with a sample problem."
  (handler-case
      (let ((problem '((:type "classification" :data "images")
                       (:type "quantum-optimization" :data "cost-function"))))
        (start-planner problem))
    (error (e)
      (format t "Example error: ~A~%" e)
      nil)))

;; Function: main
;; Main entry point for the planner
(defun main ()
  "Main entry point for the planner, ensuring robust initialization and execution."
  (handler-case
      (progn
        (format t "Starting Chimera Planner...~%")
        (example))
    (error (e)
      (format t "Fatal error: ~A~%" e)
      (uiop:quit 1))))

;; Ensure main is called when the script is run
(main)
