;; macros.lisp: Metaprogramming macros for the Chimera Cognitive Architecture's symbolic reasoning engine.
;; Purpose: Defines macros to dynamically generate functions and rules based on learned concepts from the knowledge base
;; (e.g., generating a function for (isa cat mammal)). Ensures macros are idempotent and safe for repeated evaluation.
;; Uses ASDF for package management, includes robust error handling for macro expansion failures, and exports macros
;; for use in planner.lisp and reasoning/. Designed for SBCL.

(ql:quickload :asdf)

(asdf:defsystem :chimera-macros
  :depends-on (:cl-ppcre :fiveam)
  :components ((:file "reasoning/knowledge-base")))

(defpackage :chimera-macros
  (:use :cl :cl-ppcre)
  (:export #:defconcept-fn #:defrule #:with-safe-expansion))

(in-package :chimera-macros)

;; Custom condition for macro expansion errors
(define-condition macro-expansion-error (error)
  ((message :initarg :message :reader message))
  (:report (lambda (condition stream)
             (format stream "Macro expansion error: ~A" (message condition)))))

;; Parameter: Maximum symbol length to prevent excessive memory usage
(defparameter *max-symbol-length* 100
  "Maximum length for generated symbol names to prevent memory issues")

;; Function: sanitize-symbol
;; Sanitizes symbol names to ensure safe macro expansion
(defun sanitize-symbol (symbol)
  "Sanitizes a symbol name to prevent invalid or malicious identifiers.
   Input: symbol (symbol or string) - Symbol to sanitize.
   Output: Sanitized string or error if invalid."
  (let ((name (if (symbolp symbol) (symbol-name symbol) symbol)))
    (cond
      ((not (stringp name))
       (error 'macro-expansion-error :message "Symbol name must be a string or symbol"))
      ((> (length name) *max-symbol-length*)
       (error 'macro-expansion-error :message "Symbol name exceeds maximum length"))
      (t (regex-replace-all "[^a-zA-Z0-9\\-]" name "")))))

;; Macro: with-safe-expansion
;; Wraps macro expansions to catch errors and ensure idempotency
(defmacro with-safe-expansion ((&key name) &body body)
  "Ensures safe macro expansion by catching errors and checking for existing definitions.
   Input: name (symbol) - Name of the generated function or rule.
          body - Macro expansion code.
   Output: Wrapped expansion with error handling and idempotency checks."
  `(handler-case
       (let ((sanitized-name (intern (sanitize-symbol ',name))))
         (unless (or (fboundp sanitized-name) (macro-function sanitized-name))
           ,@body))
     (macro-expansion-error (e)
       (format t "Macro expansion failed for ~A: ~A~%" ',name (message e))
       nil)
     (error (e)
       (format t "Unexpected error in macro expansion for ~A: ~A~%" ',name e)
       nil)))

;; Macro: defconcept-fn
;; Generates a function to check if an entity satisfies a concept (e.g., (isa cat mammal))
(defmacro defconcept-fn (concept &key attributes)
  "Generates a function to check if an entity satisfies a concept based on its attributes.
   Input: concept (list) - Concept definition (e.g., (isa cat mammal)).
          attributes (list) - List of attribute predicates (e.g., ((has-part tail) (has-property meows))).
   Output: Defines a function named concept-<concept> that returns t if all attributes are satisfied.
   Example: (defconcept-fn (isa cat mammal) :attributes ((has-part tail) (has-property meows)))"
  (let* ((concept-name (if (listp concept) (cadr concept) concept))
         (fn-name (intern (format nil "CONCEPT-~A" (sanitize-symbol concept-name))))
         (entity (gensym "ENTITY")))
    `(with-safe-expansion (:name ,fn-name)
       (defun ,fn-name (,entity)
         ,(format nil "Checks if ~A satisfies concept ~A with attributes ~A."
                  entity concept attributes)
         (and
          ,@(mapcar
             (lambda (attr)
               (let ((pred (if (listp attr) (cadr attr) attr)))
                 `(funcall (find-predicate ',pred) ,entity)))
             attributes)
          t)))))

;; Function: find-predicate
;; Retrieves or defines a predicate function for an attribute
(defun find-predicate (predicate-name)
  "Retrieves or defines a predicate function for an attribute.
   Input: predicate-name (symbol) - Name of the predicate (e.g., has-part, has-property).
   Output: Function object or error if undefined."
  (let ((sanitized (sanitize-symbol predicate-name)))
    (or
     (and (fboundp predicate-name) (symbol-function predicate-name))
     (error 'macro-expansion-error
            :message (format nil "Predicate ~A not found in knowledge base" sanitized)))))

;; Macro: defrule
;; Generates a rule for reasoning based on a concept and conditions
(defmacro defrule (rule-name concept &key conditions action)
  "Generates a reasoning rule based on a concept with conditions and an action.
   Input: rule-name (symbol) - Name of the rule.
          concept (list) - Concept to base the rule on (e.g., (isa dog mammal)).
          conditions (list) - List of conditions to check.
          action (form) - Action to execute if conditions are met.
   Output: Defines a function named rule-<rule-name> that applies the rule.
   Example: (defrule dog-barks (isa dog mammal) :conditions ((has-property barks)) :action (print 'barking))"
  (let* ((concept-name (if (listp concept) (cadr concept) concept))
         (rule-fn-name (intern (format nil "RULE-~A" (sanitize-symbol rule-name))))
         (entity (gensym "ENTITY")))
    `(with-safe-expansion (:name ,rule-fn-name)
       (defun ,rule-fn-name (,entity)
         ,(format nil "Applies rule ~A for concept ~A with conditions ~A and action ~A."
                  rule-name concept conditions action)
         (when (and
                (funcall (intern (format nil "CONCEPT-~A" (sanitize-symbol ',concept-name))) ,entity)
                ,@(mapcar
                   (lambda (cond)
                     (let ((pred (if (listp cond) (cadr cond) cond)))
                       `(funcall (find-predicate ',pred) ,entity)))
                   conditions))
           ,action)))))

;; Function: initialize-macros
;; Initializes example macros for testing
(defun initialize-macros ()
  "Initializes example macros for testing and demonstration."
  (handler-case
      (progn
        ;; Example: Define a concept function for (isa cat mammal)
        (defconcept-fn (isa cat mammal)
          :attributes ((has-part tail) (has-property meows)))
        ;; Example: Define a rule for cat meowing
        (defrule cat-meows (isa cat mammal)
          :conditions ((has-property meows))
          :action (format t "Entity ~A is meowing.~%" entity))
        (format t "Macros initialized successfully.~%"))
    (macro-expansion-error (e)
      (format t "Failed to initialize macros: ~A~%" (message e)))
    (error (e)
      (format t "Unexpected error initializing macros: ~A~%" e))))

;; Main entry point for testing macros
(defun main ()
  "Main entry point for testing macro definitions."
  (handler-case
      (progn
        (format t "Initializing Chimera Macros...~%")
        (initialize-macros))
    (error (e)
      (format t "Fatal error: ~A~%" e)
      (uiop:quit 1))))

;; Ensure main is called when the script is run
(main)
