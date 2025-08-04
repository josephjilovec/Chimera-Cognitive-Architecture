;; inference.lisp: Implements inference logic for the Chimera Cognitive Architecture's symbolic reasoning engine.
;; Purpose: Provides functions to reason over the knowledge base (e.g., inferring (isa dog animal) from (isa dog mammal)
;; and (isa mammal animal)). Interfaces with knowledge-base.lisp for concept retrieval and supports planner.lisp for
;; problem decomposition. Uses ASDF for package management and includes robust error handling for invalid inferences.
;; Exports functions for use in planner.lisp. Designed for SBCL.

(ql:quickload :asdf)

(asdf:defsystem :chimera-inference
  :depends-on (:chimera-knowledge-base :cl-ppcre :fiveam)
  :components ((:file "knowledge-base")))

(defpackage :chimera-inference
  (:use :cl :cl-ppcre :chimera-reasoning)
  (:export #:infer-relationships #:infer-transitive-isa #:infer-from-concept))

(in-package :chimera-inference)

;; Custom condition for inference errors
(define-condition inference-error (error)
  ((message :initarg :message :reader message))
  (:report (lambda (condition stream)
             (format stream "Inference error: ~A" (message condition)))))

;; Function: validate-concept
;; Validates a concept exists in the knowledge base
(defun validate-concept (concept)
  "Validates that a concept exists in the knowledge base.
   Input: concept (symbol or string) - Concept name to validate.
   Output: Sanitized concept name or error if invalid."
  (handler-case
      (let ((sanitized-concept (sanitize-concept concept)))
        (unless (gethash sanitized-concept *knowledge-base*)
          (error 'inference-error :message (format nil "Concept ~A not found in knowledge base" sanitized-concept)))
        sanitized-concept)
    (knowledge-base-error (e)
      (error 'inference-error :message (format nil "Validation failed: ~A" (message e))))
    (error (e)
      (error 'inference-error :message (format nil "Unexpected validation error: ~A" e)))))

;; Function: infer-transitive-isa
;; Infers transitive 'isa' relationships (e.g., dog -> mammal -> animal)
(defun infer-transitive-isa (concept target-type)
  "Infers if a concept is transitively related to a target type via 'isa' relationships.
   Input: concept (symbol or string) - Starting concept (e.g., 'dog').
          target-type (symbol or string) - Target type to check (e.g., 'animal').
   Output: t if the transitive relationship exists, nil otherwise."
  (handler-case
      (let ((sanitized-concept (validate-concept concept))
            (sanitized-target (sanitize-concept target-type))
            (visited (make-hash-table :test #'equal)))
        (labels ((traverse-isa (current)
                   (when (and current (not (gethash current visited)))
                     (setf (gethash current visited) t)
                     (let ((relationships (gethash current *knowledge-base*)))
                       (when (member sanitized-target relationships :test #'equal :key #'cadr)
                         (return-from infer-transitive-isa t))
                       (dolist (rel relationships)
                         (when (eq (car rel) 'isa)
                           (when (traverse-isa (symbol-name (cadr rel)))
                             (return-from infer-transitive-isa t))))))))
          (traverse-isa sanitized-concept)
          (format t "No transitive 'isa' relationship found from ~A to ~A~%" sanitized-concept sanitized-target)
          nil))
    (inference-error (e)
      (format t "Error in transitive inference: ~A~%" (message e))
      nil)
    (error (e)
      (format t "Unexpected error in transitive inference: ~A~%" e)
      nil)))

;; Function: infer-from-concept
;; Infers all possible relationships for a concept based on the knowledge base
(defun infer-from-concept (concept)
  "Infers all relationships for a given concept, including transitive 'isa' relationships.
   Input: concept (symbol or string) - Concept to infer relationships for (e.g., 'dog').
   Output: List of inferred relationships or nil on failure."
  (handler-case
      (let ((sanitized-concept (validate-concept concept))
            (inferred '()))
        ;; Get direct relationships
        (let ((direct-rels (gethash sanitized-concept *knowledge-base*)))
          (setf inferred (append inferred direct-rels)))
        ;; Infer transitive 'isa' relationships
        (maphash
         (lambda (key _)
           (declare (ignore _))
           (when (and (not (string= key sanitized-concept))
                      (infer-transitive-isa sanitized-concept key))
             (push `(isa ,sanitized-concept ,key) inferred)))
         *knowledge-base*)
        (if inferred
            (progn
              (format t "Inferred ~A relationships for ~A: ~A~%"
                      (length inferred) sanitized-concept inferred)
              inferred)
            (progn
              (format t "No relationships inferred for ~A~%" sanitized-concept)
              nil)))
    (inference-error (e)
      (format t "Error inferring relationships: ~A~%" (message e))
      nil)
    (error (e)
      (format t "Unexpected error inferring relationships: ~A~%" e)
      nil)))

;; Function: infer-relationships
;; Infers relationships for a list of concepts or a query pattern
(defun infer-relationships (input)
  "Infers relationships for a list of concepts or a query pattern.
   Input: input (list or string) - List of concepts or query pattern (e.g., 'mammal').
   Output: List of inferred relationships or sub-problems for planner.lisp."
  (handler-case
      (cond
        ((stringp input)
         ;; Query the knowledge base and infer relationships for matching concepts
         (let ((matches (query-knowledge-base input)))
           (when matches
             (mapcan
              (lambda (match)
                (let ((concept (getf match :concept)))
                  (infer-from-concept concept)))
              matches))))
        ((listp input)
         ;; Process each concept in the list
         (mapcan
          (lambda (concept)
            (infer-from-concept (if (listp concept) (getf concept :concept) concept)))
          input))
        (t
         (error 'inference-error :message "Input must be a string or list")))
    (inference-error (e)
      (format t "Error in infer-relationships: ~A~%" (message e))
      nil)
    (error (e)
      (format t "Unexpected error in infer-relationships: ~A~%" e)
      nil)))

;; Main entry point for testing
(defun main ()
  "Main entry point for testing the inference engine."
  (handler-case
      (progn
        (format t "Initializing Chimera Inference Engine...~%")
        (initialize-knowledge-base)
        ;; Example: Infer relationships for 'dog'
        (let ((results (infer-relationships "dog")))
          (format t "Inference results for 'dog': ~A~%" results)))
    (error (e)
      (format t "Fatal error: ~A~%" e)
      (uiop:quit 1))))

;; Ensure main is called when the script is run
(main)
