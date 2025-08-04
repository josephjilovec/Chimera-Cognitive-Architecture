;; test-planner.lisp: Unit and integration tests for the planner in backend/lisp/planner.lisp
;; using FiveAM for the Chimera Cognitive Architecture. Tests problem decomposition and task delegation
;; to the Julia layer via JSON/socket interactions. Mocks socket communication to isolate tests.
;; Aims for ≥90% test coverage of planner.lisp functions. Includes error handling tests for invalid inputs.
;; Designed for SBCL.

(ql:quickload :fiveam)
(ql:quickload :asdf)
(ql:quickload :usocket)
(ql:quickload :jsown)

(asdf:defsystem :chimera-test-planner
  :depends-on (:chimera-planner :fiveam :usocket :jsown :cl-ppcre)
  :components ((:file "backend/lisp/planner")))

(defpackage :chimera-test-planner
  (:use :cl :fiveam :chimera-planner :usocket :jsown)
  (:export #:run-tests))

(in-package :chimera-test-planner)

;; Mocking setup for Julia layer interactions
(defparameter *mock-socket* nil
  "Mock socket for simulating Julia layer communication")
(defparameter *mock-responses* (make-hash-table :test #'equal)
  "Mock responses for specific instructions")

;; Mock send-to-julia function
(defun mock-send-to-julia (instruction)
  "Mocks send-to-julia by returning pre-defined responses from *mock-responses*.
   Input: instruction - JSON string to send to Julia layer.
   Output: Mocked JSON response or error."
  (let ((response (gethash instruction *mock-responses*)))
    (if response
        (progn
          (format t "Mock response for ~A: ~A~%" instruction response)
          response)
        (error 'invalid-input-error :message (format nil "No mock response for instruction: ~A" instruction)))))

;; Setup mock responses
(defun setup-mock-responses ()
  "Sets up mock responses for common instructions."
  (clrhash *mock-responses*)
  (setf (gethash "{\"module\":\"Flux\",\"code\":\"generate_network({'layers': [{'type': 'dense', 'input_dim': 784, 'output_dim': 128, 'activation': 'relu'}]})\"}" *mock-responses*)
        "{\"status\": \"success\", \"message\": \"Network created\", \"network\": \"Chain(Dense(784 => 128))\"}")
  (setf (gethash "{\"module\":\"Yao\",\"code\":\"generate_circuit({'n_qubits': 2, 'gates': [{'type': 'H', 'qubits': [1]}]})\"}" *mock-responses*)
        "{\"status\": \"success\", \"results\": {\"00\": 50, \"01\": 50}, \"n_shots\": 100}")
  (setf (gethash "{\"module\":\"CUDA\",\"code\":\"execute_gpu_computation({...})\"}" *mock-responses*)
        "{\"status\": \"success\", \"output\": [0.1, 0.9], \"loss\": 0.05}")
  t)

;; Test suite definition
(def-suite planner-suite
  :description "Test suite for the Chimera planner")

(in-suite planner-suite)

;; Test: decompose-problem
(test test-decompose-problem
  "Tests problem decomposition into sub-problems for neural and quantum tasks."
  (let ((*knowledge-base* (make-hash-table :test #'equal)))
    ;; Mock knowledge base data
    (setf (gethash "classification" *knowledge-base*)
          '((isa task) (has-module Flux) (has-data images)))
    (setf (gethash "quantum-optimization" *knowledge-base*)
          '((isa task) (has-module Yao) (has-data cost-function)))
    (let ((result (decompose-problem "classification")))
      (is-true (listp result)
               "Decomposed problem should return a list")
      (is (find-if (lambda (sub) (eq (getf sub :module) :Flux)) result)
          "Classification problem should include Flux sub-problem"))
    (let ((result (decompose-problem "quantum-optimization")))
      (is (find-if (lambda (sub) (eq (getf sub :module) :Yao)) result)
          "Quantum optimization problem should include Yao sub-problem"))
    (signals invalid-input-error (decompose-problem nil)
             "Nil problem should signal invalid-input-error")
    (signals invalid-input-error (decompose-problem "unknown")
             "Unknown problem should signal invalid-input-error")))

;; Test: generate-instructions
(test test-generate-instructions
  "Tests generation of JSON instructions for sub-problems."
  (let ((sub-problems (list
                       (list :module :Flux :task "classification" :params (list :layers '((dense 784 128 :relu))))
                       (list :module :Yao :task "quantum-optimization" :params (list :n_qubits 2 :gates '((H 1)))))))
    (let ((instructions (generate-instructions sub-problems)))
      (is-true (listp instructions)
               "Instructions should be a list")
      (is (= (length instructions) 2)
          "Should generate two instructions")
      (is (find-if (lambda (inst) (string= (getf (jsown:parse inst) :module) "Flux")) instructions)
          "Should include Flux instruction")
      (is (find-if (lambda (inst) (string= (getf (jsown:parse inst) :module) "Yao")) instructions)
          "Should include Yao instruction"))
    (signals invalid-input-error (generate-in ACTIONS nil)
             "Nil sub-problems should signal invalid-input-error")
    (signals invalid-input-error (generate-instructions (list (list :module :Unknown)))
             "Unknown module should signal invalid-input-error")))

;; Test: send-to-julia
(test test-send-to-julia
  "Tests sending instructions to the Julia layer using mock responses."
  (let ((*julia-host* "localhost")
        (*julia-port* 5000))
    (setup-mock-responses)
    (let ((instruction "{\"module\":\"Flux\",\"code\":\"generate_network({'layers': [{'type': 'dense', 'input_dim': 784, 'output_dim': 128, 'activation': 'relu'}]})\"}")
          (*send-to-julia* #'mock-send-to-julia))
      (let ((response (send-to-julia instruction)))
        (is-true (stringp response)
                 "Response should be a JSON string")
        (is (string= (getf (jsown:parse response) :status) "success")
            "Flux instruction should return success")))
    (let ((instruction "{\"module\":\"Yao\",\"code\":\"generate_circuit({'n_qubits': 2, 'gates': [{'type': 'H', 'qubits': [1]}]})\"}")
          (*send-to-julia* #'mock-send-to-julia))
      (let ((response (send-to-julia instruction)))
        (is (string= (getf (jsown:parse response) :status) "success")
            "Yao instruction should return success")))
    (signals invalid-input-error (send-to-julia nil)
             "Nil instruction should signal invalid-input-error")
    (signals invalid-input-error (send-to-julia "{\"module\":\"Unknown\"}")
             "Unknown module should signal invalid-input-error")))

;; Test: validate-task
(test test-validate-task
  "Tests validation of tasks before delegation."
  (let ((*knowledge-base* (make-hash-table :test #'equal)))
    (setf (gethash "classification" *knowledge-base*)
          '((isa task) (has-module Flux) (has-data images)))
    (is-true (validate-task "classification")
             "Valid task should return t")
    (signals invalid-input-error (validate-task nil)
             "Nil task should signal invalid-input
