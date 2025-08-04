;; knowledge-base.lisp: Implements the symbolic knowledge base for the Chimera Cognitive Architecture's reasoning engine.
;; Purpose: Stores concepts and relationships (e.g., (isa dog mammal), (has-part dog tail)) using hash tables for
;; efficient retrieval. Provides functions to query and update the knowledge base, ensuring consistency and integration
;; with planner.lisp and macros.lisp. Uses ASDF for package management and includes robust error handling for
;; inconsistent or missing concepts. Designed for SBCL.

(ql:quickload :asdf)

(asdf:defsystem :chimera-knowledge-base
  :depends-on (:cl-ppcre :fiveam)
  :components ())

(defpackage :chimera-reasoning
  (:use :cl :cl-ppcre)
  (:export #:initialize-knowledge-base #:add-concept #:add-relationship #:query-knowledge-base
           #:find-predicate #:update-concept #:remove-concept))

(in-package :chimera-reasoning)

;; Configuration parameters
(defparameter *knowledge-base* (make-hash-table :test #'equal)
  "Hash table storing concepts and relationships (e.g., 'dog' -> ((isa mammal) (has-part tail)))")
(defparameter *predicates* (make-hash-table :test #'equal)
  "Hash table storing predicate functions (e.g., 'has-part' -> function)")
(defparameter *max-concept-length* 100
  "Maximum length for concept names to prevent memory issues")

;; Custom condition for knowledge base errors
(define-condition knowledge-base-error (error)
  ((message :initarg :message :reader message))
  (:report (lambda (condition stream)
             (format stream "Knowledge base error: ~A" (message condition)))))

;; Function: sanitize-concept
;; Sanitizes concept names to ensure safe storage and retrieval
(defun sanitize-concept (concept)
  "Sanitizes a concept name to prevent invalid or malicious identifiers.
   Input: concept (symbol or string) - Concept name to sanitize.
   Output: Sanitized string or error if invalid."
  (let ((name (if (symbolp concept) (symbol-name concept) concept)))
    (cond
      ((not (stringp name))
       (error 'knowledge-base-error :message "Concept name must be a string or symbol"))
      ((> (length name) *max-concept-length*)
       (error 'knowledge-base-error :message "Concept name exceeds maximum length"))
      (t (regex-replace-all "[^a-zA-Z0-9\\-]" name "")))))

;; Function: initialize-knowledge-base
;; Initializes the knowledge base with default concepts and predicates
(defun initialize-knowledge-base ()
  "Initializes the knowledge base with default concepts and predicates.
   Output: Hash table containing the knowledge base."
  (handler-case
      (progn
        ;; Clear existing knowledge base
        (clrhash *knowledge-base*)
        (clrhash *predicates*)
        ;; Define default predicates
        (setf (gethash "has-part" *predicates*)
              (lambda (entity part) (declare (ignore entity part)) t)) ; Placeholder predicate
        (setf (gethash "has-property" *predicates*)
              (lambda (entity prop) (declare (ignore entity prop)) t)) ; Placeholder predicate
        ;; Add example concepts
        (add-concept "dog" '((isa mammal) (has-part tail) (has-property barks)))
        (add-concept "cat" '((isa mammal) (has-part tail) (has-property meows)))
        (format t "Knowledge base initialized with default concepts.~%")
        *knowledge-base*)
    (error (e)
      (error 'knowledge-base-error :message (format nil "Failed to initialize knowledge base: ~A" e)))))

;; Function: add-concept
;; Adds a new concept with its relationships to the knowledge base
(defun add-concept (concept relationships)
  "Adds a concept and its relationships to the knowledge base.
   Input: concept (symbol or string) - Concept name (e.g., 'dog').
          relationships (list) - List of relationships (e.g., ((isa mammal) (has-part tail))).
   Output: t if successful, error if invalid."
  (handler-case
      (let ((sanitized-concept (sanitize-concept concept)))
        (unless (and (listp relationships)
                     (every (lambda (rel) (and (listp rel) (>= (length rel) 2))) relationships))
          (error 'knowledge-base-error :message "Relationships must be a list of valid relation pairs"))
        (dolist (rel relationships)
          (unless (gethash (symbol-name (car rel)) *predicates*)
            (error 'knowledge-base-error :message (format nil "Unknown predicate: ~A" (car rel)))))
        (setf (gethash sanitized-concept *knowledge-base*) relationships)
        (format t "Added concept ~A with relationships ~A~%" sanitized-concept relationships)
        t)
    (knowledge-base-error (e)
      (format t "Error adding concept: ~A~%" (message e))
      nil)
    (error (e)
      (format t "Unexpected error adding concept: ~A~%" e)
      nil)))

;; Function: add-relationship
;; Adds a relationship to an existing concept
(defun add-relationship (concept relationship)
  "Adds a relationship to an existing concept in the knowledge base.
   Input: concept (symbol or string) - Concept name.
          relationship (list) - Single relationship (e.g., (has-part tail)).
   Output: t if successful, error if invalid."
  (handler-case
      (let ((sanitized-concept (sanitize-concept concept)))
        (unless (gethash sanitized-concept *knowledge-base*)
          (error 'knowledge-base-error :message (format nil "Concept ~A not found" sanitized-concept)))
        (unless (and (listp relationship) (>= (length relationship) 2))
          (error 'knowledge-base-error :message "Relationship must be a valid relation pair"))
        (unless (gethash (symbol-name (car relationship)) *predicates*)
          (error 'knowledge-base-error :message (format nil "Unknown predicate: ~A" (car relationship))))
        (push relationship (gethash sanitized-concept *knowledge-base*))
        (format t "Added relationship ~A to concept ~A~%" relationship sanitized-concept)
        t)
    (knowledge-base-error (e)
      (format t "Error adding relationship: ~A~%" (message e))
      nil)
    (error (e)
      (format t "Unexpected error adding relationship: ~A~%" e)
      nil)))

;; Function: query-knowledge-base
;; Queries the knowledge base for concepts matching a pattern
(defun query-knowledge-base (pattern)
  "Queries the knowledge base for concepts matching a pattern.
   Input: pattern (string) - Search pattern (e.g., 'mammal' to find (isa ? mammal)).
   Output: List of matching concepts and their relationships."
  (handler-case
      (let ((sanitized-pattern (sanitize-concept pattern))
            (results nil))
        (maphash
         (lambda (concept relationships)
           (when (or (search sanitized-pattern concept :test #'char-equal)
                     (some (lambda (rel)
                             (and (eq (car rel) 'isa)
                                  (search sanitized-pattern (symbol-name (cadr rel)) :test #'char-equal)))
                           relationships))
             (push (list :concept concept :relationships relationships) results)))
         *knowledge-base*)
        (if results
            (progn
              (format t "Found ~A matching concepts for pattern ~A~%" (length results) sanitized-pattern)
              results)
            (progn
              (format t "No concepts found for pattern ~A~%" sanitized-pattern)
              nil)))
    (knowledge-base-error (e)
      (format t "Error querying knowledge base: ~A~%" (message e))
      nil)
    (error (e)
      (format t "Unexpected error querying knowledge base: ~A~%" e)
      nil)))

;; Function: find-predicate
;; Retrieves a predicate function from the predicates hash table
(defun find-predicate (predicate-name)
  "Retrieves a predicate function by name.
   Input: predicate-name (symbol or string) - Predicate name (e.g., 'has-part').
   Output: Function object or error if not found."
  (handler-case
      (let ((sanitized-name (sanitize-concept predicate-name)))
        (or (gethash sanitized-name *predicates*)
            (error 'knowledge-base-error :message (format nil "Predicate ~A not found" sanitized-name))))
    (knowledge-base-error (e)
      (format t "Error finding predicate: ~A~%" (message e))
      nil)
    (error (e)
      (format t "Unexpected error finding predicate: ~A~%" e)
      nil)))

;; Function: update-concept
;; Updates relationships for an existing concept
(defun update-concept (concept new-relationships)
  "Updates the relationships for an existing concept.
   Input: concept (symbol or string) - Concept name.
          new-relationships (list) - New list of relationships.
   Output: t if successful, error if invalid."
  (handler-case
      (let ((sanitized-concept (sanitize-concept concept)))
        (unless (gethash sanitized-concept *knowledge-base*)
          (error 'knowledge-base-error :message (format nil "Concept ~A not found" sanitized-concept)))
        (unless (and (listp new-relationships)
                     (every (lambda (rel) (and (listp rel) (>= (length rel) 2))) new-relationships))
          (error 'knowledge-base-error :message "New relationships must be a list of valid relation pairs"))
        (dolist (rel new-relationships)
          (unless (gethash (symbol-name (car rel)) *predicates*)
            (error 'knowledge-base-error :message (format nil "Unknown predicate: ~A" (car rel)))))
        (setf (gethash sanitized-concept *knowledge-base*) new-relationships)
        (format t "Updated concept ~A with new relationships ~A~%" sanitized-concept new-relationships)
        t)
    (knowledge-base-error (e)
      (format t "Error updating concept: ~A~%" (message e))
      nil)
    (error (e)
      (format t "Unexpected error updating concept: ~A~%" e)
      nil)))

;; Function: remove-concept
;; Removes a concept from the knowledge base
(defun remove-concept (concept)
  "Removes a concept from the knowledge base.
   Input: concept (symbol or string) - Concept name.
   Output: t if successful, error if not found."
  (handler-case
      (let ((sanitized-concept (sanitize-concept concept)))
        (unless (gethash sanitized-concept *knowledge-base*)
          (error 'knowledge-base-error :message (format nil "Concept ~A not found" sanitized-concept)))
        (remhash sanitized-concept *knowledge-base*)
        (format t "Removed concept ~A from knowledge base~%" sanitized-concept)
        t)
    (knowledge-base-error (e)
      (format t "Error removing concept: ~A~%" (message e))
      nil)
    (error (e)
      (format t "Unexpected error removing concept: ~A~%" e)
      nil)))

;; Main entry point for testing
(defun main ()
  "Main entry point for testing the knowledge base."
  (handler-case
      (progn
        (format t "Initializing Chimera Knowledge Base...~%")
        (initialize-knowledge-base)
        ;; Example: Query for mammal-related concepts
        (query-knowledge-base "mammal"))
    (error (e)
      (format t "Fatal error: ~A~%" e)
      (uiop:quit 1))))

;; Ensure main is called when the script is run
(main)
